"""Private server-to-server client for the speaking assessment service."""

import re
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, UploadFile


_ID = re.compile(r"[0-9a-f]{32}\Z")
_USER_ERRORS = {400, 404, 422, 429, 503}


class SpeakingService:
    def __init__(self, base_url: str, admin_password: str = "", client: httpx.Client | None = None):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or not parsed.port
            or parsed.path not in {"", "/"}
            or parsed.query or parsed.fragment or parsed.username or parsed.password
        ):
            raise ValueError("The speaking service must use an explicit loopback HTTP origin and port.")
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(connect=5, read=180, write=30, pool=5),
            trust_env=False,
        )
        self.admin_password = admin_password

    def close(self):
        self.client.close()

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self.client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise HTTPException(503, "Speaking assessment is unavailable. Try again later.") from exc
        if response.status_code >= 400:
            if response.status_code in _USER_ERRORS:
                try:
                    detail = response.json().get("detail")
                except (ValueError, AttributeError):
                    detail = None
                if not isinstance(detail, str):
                    detail = "Speaking assessment could not complete this request."
                raise HTTPException(response.status_code, detail)
            raise HTTPException(502, "Speaking assessment returned an unexpected error.")
        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(502, "Speaking assessment returned an invalid response.") from exc

    def create_speaker(self, site_student_id: str) -> str:
        if not _ID.fullmatch(site_student_id):
            raise ValueError("Invalid website learner ID.")
        name = f"site-{site_student_id}"
        profile = self._request("POST", "/api/students", json={"name": name, "group_type": "foreigner"})
        if not isinstance(profile, dict) or profile.get("name") != name or not _ID.fullmatch(str(profile.get("id", ""))):
            raise HTTPException(502, "Speaking assessment returned an invalid learner profile.")
        return profile["id"]

    def calibrate(self, speaker_id: str, file: UploadFile):
        self._valid_id(speaker_id)
        return self._upload("POST", f"/api/students/{speaker_id}/voice-calibration", file)

    def assess(self, speaker_id: str, file: UploadFile, *, script: str, sample_id: str,
               practice_mode: str, scoring_mode: str, save_attempt: bool):
        self._valid_id(speaker_id)
        if not save_attempt:
            try:
                capabilities = self._request("GET", "/api/speaking-capabilities")
            except HTTPException as exc:
                if exc.status_code == 404:
                    raise HTTPException(503, "Speaking assessor needs the stateless assessment update.") from exc
                raise
            if not isinstance(capabilities, dict) or capabilities.get("stateless_assessment") is not True:
                raise HTTPException(503, "Speaking assessor cannot score without saving a second attempt.")
        return self._upload("POST", "/api/assess-pronunciation", file, data={
            "script": script, "sample_id": sample_id, "student_id": speaker_id,
            "practice_mode": practice_mode, "scoring_mode": scoring_mode,
            "save_attempt": "true" if save_attempt else "false",
        })

    def delete_speaker(self, speaker_id: str):
        self._valid_id(speaker_id)
        auth = httpx.BasicAuth("admin", self.admin_password) if self.admin_password else None
        try:
            self._request("DELETE", f"/api/admin/students/{speaker_id}", auth=auth)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise

    def _upload(self, method: str, path: str, file: UploadFile, data: dict | None = None):
        file.file.seek(0)
        try:
            return self._request(method, path, data=data, files={
                "file": (file.filename or "recording.webm", file.file, file.content_type or "application/octet-stream")
            })
        finally:
            file.file.seek(0)

    @staticmethod
    def _valid_id(value: str):
        if not _ID.fullmatch(value):
            raise ValueError("Invalid speaking learner ID.")
