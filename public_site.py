"""Authenticated React learning site. Installed only in MANDARIN_PUBLIC_MODE."""
import io
import json
import os
import re
import secrets
import shutil
import tempfile
import zipfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from public_curriculum import questions, stories
from public_store import PublicStore

COOKIE = 'mandarin_public_session'
LOCALES = {'en', 'zh-TW', 'vi', 'id', 'hi'}


class ProfileInput(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)
    locale: str = 'vi'
    study_goal: str = Field(default='', max_length=180)
    weekly_days: int = Field(default=3, ge=1, le=7)
    leaderboard_opt_in: bool = False


class QuizInput(BaseModel):
    story_id: str
    skill: str
    answers: dict[str, str]


def install_public_site(app, *, samples, words, meanings, practice, coordinator, assess, calibrate, pinyin,
                        speaking_service=None):
    store = PublicStore(practice)
    secure = os.getenv('MANDARIN_PUBLIC_HTTPS', '0') == '1'
    dev_auth = os.getenv('MANDARIN_DEV_AUTH', '0') == '1' and not secure
    if secure:
        required = ('MANDARIN_SESSION_SECRET', 'MANDARIN_PUBLIC_ORIGIN', 'GOOGLE_CLIENT_ID',
                    'GOOGLE_CLIENT_SECRET', 'MANDARIN_ASSESSOR_ADMIN_PASSWORD')
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise RuntimeError('Public HTTPS configuration is incomplete: ' + ', '.join(missing))
        if not os.environ['MANDARIN_PUBLIC_ORIGIN'].startswith('https://'):
            raise RuntimeError('MANDARIN_PUBLIC_ORIGIN must be an HTTPS origin.')
    secret = os.getenv('MANDARIN_SESSION_SECRET') or secrets.token_urlsafe(40)
    app.add_middleware(SessionMiddleware, secret_key=secret, https_only=secure, same_site='lax')
    oauth = OAuth()
    if os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET'):
        oauth.register('google', client_id=os.getenv('GOOGLE_CLIENT_ID'), client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'})
    router = APIRouter()
    grouped = stories(samples)

    def shown_pinyin(script):
        tokens = pinyin(script)
        if isinstance(tokens, str):
            return tokens
        return re.sub(r'\s+([，。！？、；：])', r'\1', ' '.join(tokens))

    def current(request: Request):
        account = store.session(request.cookies.get(COOKIE))
        if not account:
            raise HTTPException(401, 'Sign in to continue.')
        return account

    def mutation(request: Request, account=Depends(current)):
        origin = request.headers.get('origin')
        if origin and origin != f'{request.url.scheme}://{request.headers.get("host")}' and origin != os.getenv('MANDARIN_PUBLIC_ORIGIN'):
            raise HTTPException(403, 'Invalid request origin.')
        if not secrets.compare_digest(request.headers.get('x-csrf-token', ''), account['csrf_token']):
            raise HTTPException(403, 'Invalid CSRF token.')
        return account

    def visible(account):
        return {key: account[key] for key in ('display_name','avatar_url','locale','study_goal','weekly_days','leaderboard_opt_in','onboarding_done')}

    def speaker_for(account):
        speaker_id = store.speaking_profile(account['student_id'])
        if speaker_id is None:
            speaker_id = speaking_service.create_speaker(account['student_id'])
            store.bind_speaking_profile(account['student_id'], speaker_id)
        return speaker_id

    @router.get('/api/public/config')
    def config():
        return {'google_enabled': bool(oauth.create_client('google')), 'dev_auth': dev_auth}

    @router.get('/auth/google/start')
    async def google_start(request: Request):
        client = oauth.create_client('google')
        if not client:
            raise HTTPException(503, 'Google login is not configured.')
        return await client.authorize_redirect(request, str(request.url_for('google_callback')))

    @router.get('/auth/google/callback', name='google_callback')
    async def google_callback(request: Request):
        client = oauth.create_client('google')
        if not client:
            raise HTTPException(503, 'Google login is not configured.')
        try:
            token = await client.authorize_access_token(request)
            identity = token['userinfo']  # Authlib validates state, nonce and ID token.
        except Exception:
            raise HTTPException(401, 'Google sign-in could not be verified.')
        sub = str(identity.get('sub') or '')
        if not sub:
            raise HTTPException(401, 'Google identity is incomplete.')
        account = store.account_for_google(sub, identity.get('email',''), identity.get('name',''), identity.get('picture',''))
        response = RedirectResponse('/profile' if account['onboarding_done'] else '/welcome', status_code=303)
        response.set_cookie(COOKIE, store.create_session(account['id']), max_age=14*86400, httponly=True, secure=secure, samesite='lax', path='/')
        request.session.clear()
        return response

    @router.post('/auth/logout')
    def logout(request: Request, account=Depends(mutation)):
        store.revoke(request.cookies.get(COOKIE))
        response = JSONResponse({'ok': True})
        response.delete_cookie(COOKIE, path='/')
        return response

    @router.post('/auth/dev-login')
    def dev_login(request: Request):
        if not dev_auth or request.client.host not in {'127.0.0.1','::1','localhost','testclient'} or request.url.hostname not in {'127.0.0.1','::1','localhost','testserver'}:
            raise HTTPException(404)
        account = store.account_for_google('local-dev-user', '', 'Người học thử', '')
        response = JSONResponse({'ok': True})
        response.set_cookie(COOKIE, store.create_session(account['id']), max_age=14*86400, httponly=True, secure=secure, samesite='lax', path='/')
        return response

    @router.get('/api/me/session')
    def session(account=Depends(current)):
        return {'profile': visible(account), 'csrf_token': account['csrf_token']}

    @router.get('/api/me/profile')
    def profile(account=Depends(current)):
        return visible(account)

    @router.patch('/api/me/profile')
    def update_profile(payload: ProfileInput, account=Depends(mutation)):
        if payload.locale not in LOCALES:
            raise HTTPException(422, 'Unsupported language.')
        values = payload.model_dump()
        values['display_name'] = values['display_name'].strip()
        if not values['display_name']:
            raise HTTPException(422, 'Enter a display name.')
        values['study_goal'] = values['study_goal'].strip()
        values['onboarding_done'] = 1
        return visible(store.update_profile(account['student_id'], values))

    @router.get('/api/public/catalog')
    def catalog():
        lessons = []
        for lesson in sorted({row['lesson'] for row in samples.values()}):
            lesson_stories = []
            for story_id, rows in grouped.items():
                if rows[0]['lesson'] == lesson:
                    lesson_stories.append({'id': story_id, 'story': rows[0]['story'], 'scenes': len(rows), 'audio_available': all(row['audio_available'] for row in rows)})
            lessons.append({'id': lesson, 'stories': lesson_stories})
        return {'collections': [{'id':'textbook','lessons':lessons}, {'id':'chapter5-words','lessons':[{'id':5,'items':len(words)}]}]}

    @router.get('/api/me/stories/{story_id}/{skill}')
    def exercise(story_id: str, skill: str, account=Depends(current)):
        rows = grouped.get(story_id)
        if not rows or skill not in ('listen','read','speak'):
            raise HTTPException(404)
        locale = account['locale']
        from public_curriculum import translated
        scenes = [{'id':r['id'], 'scene':r['scene'], 'script':r['script'] if skill != 'listen' else None,
                   'pinyin':shown_pinyin(r['script']) if skill == 'speak' else None,
                   'meaning':translated(r,locale,meanings) if skill == 'speak' else None,
                   'audio_url':f"/api/me/samples/{r['id']}/audio" if r['audio_available'] and skill != 'read' else None}
                  for r in rows]
        qs = questions(story_id, skill, locale, samples, meanings) if skill != 'speak' else []
        return {'id':story_id, 'lesson':rows[0]['lesson'], 'story':rows[0]['story'], 'skill':skill,
                'scenes':scenes, 'questions':[{k:v for k,v in q.items() if k not in ('answer','explanation')} for q in qs]}

    @router.get('/api/me/story-reviews/{story_id}')
    def review(story_id: str, skill: str, account=Depends(current)):
        rows = grouped.get(story_id)
        if not rows or skill not in ('listen', 'read'):
            raise HTTPException(404)
        if not any(r['story_id']==story_id and r['skill']==skill for r in store.quiz_rows(account['student_id'])):
            raise HTTPException(403, 'Submit an exercise first.')
        return [{'scene':r['scene'],'script':r['script'],'pinyin':shown_pinyin(r['script']),
                 'meaning':__import__('public_curriculum').translated(r, account['locale'], meanings),
                 'audio_url':f"/api/me/samples/{r['id']}/audio" if r['audio_available'] else None} for r in rows]

    @router.post('/api/me/exercise-attempts')
    def submit_quiz(payload: QuizInput, account=Depends(mutation)):
        qs = questions(payload.story_id, payload.skill, account['locale'], samples, meanings)
        if not qs:
            raise HTTPException(404)
        expected = {q['id'] for q in qs}
        if set(payload.answers) != expected:
            raise HTTPException(422, 'Answer all three questions.')
        correct = sum(payload.answers[q['id']] == q['answer'] for q in qs)
        attempt_id = store.save_quiz(account['student_id'], payload.story_id, payload.skill, payload.answers, correct, len(qs))
        return {'id':attempt_id,'correct':correct,'total':len(qs), 'review':[
            {'id':q['id'],'correct':payload.answers[q['id']]==q['answer'], 'answer':q['answer'], 'explanation':q['explanation']}
            for q in qs]}

    @router.get('/api/me/samples/{sample_id}/audio')
    def sample_audio(sample_id: str, account=Depends(current)):
        sample = samples.get(sample_id)
        if not sample or not Path(sample['audio_path']).is_file():
            raise HTTPException(404)
        return FileResponse(sample['audio_path'], media_type='audio/mpeg')

    @router.get('/api/me/words')
    def word_catalog(account=Depends(current)):
        return [{'id':row['word_key'],'script':row['script'],'pinyin':row['pinyin'],
                 'meaning':row.get('english_meaning',''),'category':row.get('section',''),
                 'audio_url':f"/api/me/words/{row['word_key']}/audio"} for row in words.values()]

    @router.get('/api/me/words/{word_id}/audio')
    def word_audio(word_id: str, account=Depends(current)):
        item = words.get(word_id)
        if not item or not Path(item['audio_path']).is_file():
            raise HTTPException(404)
        return FileResponse(item['audio_path'], media_type='audio/mpeg')

    @router.post('/api/me/voice-calibration')
    def voice_calibration(file: UploadFile = File(...), account=Depends(mutation)):
        with coordinator.assessment(account['student_id']):
            if speaking_service is not None:
                result = speaking_service.calibrate(speaker_for(account), file)
                calibration = result.get('voice_calibration') if isinstance(result, dict) else None
                if not isinstance(calibration, dict):
                    raise HTTPException(502, 'Speaking assessment returned an invalid calibration.')
                practice.save_voice_calibration(account['student_id'], calibration)
                return result
            return calibrate(account['student_id'], file)

    @router.get('/api/me/calibration-sound/{tone}')
    def calibration_sound(tone: int, account=Depends(current)):
        path = Path(__file__).resolve().parent / 'assets' / 'calibration' / f'ma{tone}.wav'
        if tone not in (1, 2, 3, 4, 5) or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type='audio/wav')

    @router.post('/api/me/speaking/assess')
    def speaking_assess(file: UploadFile = File(...), sample_id: str = Form(...),
                        practice_mode: str = Form('book'), scoring_mode: str = Form('fusion'), account=Depends(mutation)):
        if practice_mode == 'book':
            item = samples.get(sample_id)
        elif practice_mode == 'chapter5_word':
            item = words.get(sample_id)
        else:
            item = None
        if not item:
            raise HTTPException(404, 'Practice item not found.')
        with coordinator.assessment(account['student_id']):
            if speaking_service is not None:
                result = speaking_service.assess(speaker_for(account), file, script=item['script'],
                    sample_id=sample_id, practice_mode=practice_mode, scoring_mode=scoring_mode)
                if not isinstance(result, dict) or not isinstance(result.get('summary'), dict):
                    raise HTTPException(502, 'Speaking assessment returned an invalid result.')
                # The existing assessor always saves its own attempt. Keep the
                # website's copy linked only to the signed-in website account.
                result.pop('saved_attempt', None)
                filename = Path(file.filename or 'recording.webm').name
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temporary:
                    file.file.seek(0)
                    shutil.copyfileobj(file.file, temporary)
                    temp_path = Path(temporary.name)
                try:
                    result['saved_attempt'] = practice.save_attempt(
                        account['student_id'], sample_id, item['script'], temp_path, filename, result)
                finally:
                    temp_path.unlink(missing_ok=True)
                return result
            return assess(file, item['script'], sample_id, account['student_id'], practice_mode, '', scoring_mode)

    @router.get('/api/me/attempts/{attempt_id}/audio')
    def own_audio(attempt_id: str, account=Depends(current)):
        if not store.owns_audio(account['student_id'], attempt_id):
            raise HTTPException(404)
        path = practice.get_attempt_audio_path(attempt_id)
        if not path:
            raise HTTPException(404)
        return FileResponse(path)

    @router.get('/api/me/activity')
    def activity(account=Depends(current)):
        quizzes = [{'skill':r['skill'],'story_id':r['story_id'],'score':r['correct'],'total':r['total'],'created_at':r['created_at']} for r in store.quiz_rows(account['student_id'])]
        speaking = [{'skill':'speak','sample_id':r['sample_id'],'created_at':r['created_at']} for r in store.speaking_rows(account['student_id'])]
        return sorted(quizzes + speaking, key=lambda r:r['created_at'], reverse=True)[:50]

    @router.get('/api/me/dashboard')
    def dashboard(account=Depends(current)):
        quiz = store.quiz_rows(account['student_id'])
        spoken = practice.progress(account['student_id'])['lessons']
        progress = {}
        for skill in ('listen','read'):
            rows = [r for r in quiz if r['skill']==skill]
            progress[skill] = {'completed':len({r['story_id'] for r in rows}),'total':len(grouped),
                'last_score':round(100*rows[0]['correct']/rows[0]['total']) if rows else None,
                'best_score':max((round(100*r['correct']/r['total']) for r in rows),default=None)}
        progress['speak'] = {'completed':len(spoken),'total':len(samples)+len(words),
            'last_score':next((r['last'] for r in spoken.values() if r.get('last_at')==max((v.get('last_at','') for v in spoken.values()),default='')),None),
            'best_score':max((r['best'] for r in spoken.values()),default=None)}
        all_dates = [r['created_at'] for r in quiz] + [r['created_at'] for r in store.speaking_rows(account['student_id'])]
        taipei = ZoneInfo('Asia/Taipei')
        dates = {datetime.fromisoformat(s).astimezone(taipei).date() for s in all_dates}
        today = datetime.now(taipei).date()
        monday = today.fromordinal(today.toordinal()-today.weekday())
        days = sum(monday <= day <= today for day in dates)
        latest = sorted(quiz + store.speaking_rows(account['student_id']), key=lambda r:r['created_at'], reverse=True)
        next_story = next((key for key in grouped if key not in {r['story_id'] for r in quiz}), next(iter(grouped), None))
        analytics = practice.analytics(account['student_id']) if spoken else None
        preview = grouped[next_story][0] if next_story else None
        return {'profile':visible(account),'progress':progress,'weekly':{'active_days':days,'target':account['weekly_days']},
                'continue_story':next_story,'continue_preview':{'script':preview['script'],'pinyin':shown_pinyin(preview['script'])} if preview else None,'recent':[
                    {'skill':r.get('skill','speak'),'story_id':r.get('story_id'), 'sample_id':r.get('sample_id'),
                     'created_at':r['created_at'], 'score':round(100*r['correct']/r['total']) if 'correct' in r else None} for r in latest[:6]],
                'pronunciation':{'weakest_tone':analytics.get('weakest_tone'), 'recommendations':analytics.get('recommended_sample_ids',[])} if analytics else None,
                'achievements':{'first_listen':progress['listen']['completed']>0,'first_read':progress['read']['completed']>0,
                                'first_speak':progress['speak']['completed']>0,'weekly_goal':days>=account['weekly_days']}}

    @router.get('/api/me/leaderboard')
    def leaderboard(account=Depends(current)):
        with store.db() as db:
            rows = db.execute("""SELECT a.student_id,a.display_name,p.sample_id,p.summary_json FROM public_accounts a
                JOIN practice_attempts p ON p.student_id=a.student_id WHERE a.leaderboard_opt_in=1 AND p.sample_id<>'custom'""").fetchall()
        best = {}
        names = {}
        for row in rows:
            summary = json.loads(row['summary_json'] or '{}')
            if summary.get('scoring_mode') == 'model_voice':
                continue
            counts = summary.get('syllables') or {}
            good = counts.get('likely_correct',0) or 0
            total = sum(counts.get(k,0) or 0 for k in ('likely_correct','insufficient_evidence','practice'))
            if not good or not total:
                continue
            key = (row['student_id'],row['sample_id'])
            best[key] = max(best.get(key,0),round(100*good/total))
            names[row['student_id']] = row['display_name']
        totals = {}
        for (student,_),points in best.items():
            totals[student] = totals.get(student,0)+points
        ranked = sorted(totals, key=lambda s:-totals[s])
        return {'top':[{'rank':i+1,'name':names[s],'points':totals[s], 'is_me':s==account['student_id']} for i,s in enumerate(ranked[:10])],
                'me':next(({'rank':i+1,'points':totals[s]} for i,s in enumerate(ranked) if s==account['student_id']),None)}

    @router.get('/api/me/export')
    def export(account=Depends(current)):
        with store.db() as db:
            speaking = [dict(r) for r in db.execute("SELECT id,sample_id,result_json,audio_path,created_at FROM practice_attempts WHERE student_id=?", (account['student_id'],))]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('profile.json',json.dumps(visible(account),ensure_ascii=False,indent=2))
            archive.writestr('comprehension.json',json.dumps(store.quiz_rows(account['student_id']),ensure_ascii=False,indent=2))
            archive.writestr('speaking.json',json.dumps([{k:v for k,v in r.items() if k!='audio_path'} for r in speaking],ensure_ascii=False,indent=2))
            calibration = practice.get_voice_calibration(account['student_id'])
            if calibration:
                archive.writestr('voice_calibration.json',json.dumps(calibration,ensure_ascii=False,indent=2))
            for row in speaking:
                path = practice.get_attempt_audio_path(row['id'])
                if path and store.owns_audio(account['student_id'],row['id']) and Path(path).is_file():
                    archive.write(path,f"recordings/{row['id']}{path.suffix}")
        buffer.seek(0)
        return StreamingResponse(buffer,media_type='application/zip',headers={'Content-Disposition':'attachment; filename="mandarin-data.zip"'})

    @router.delete('/api/me/profile')
    def delete_profile(account=Depends(mutation)):
        with coordinator.student(account['student_id']) if coordinator else nullcontext():
            speaker_id = store.speaking_profile(account['student_id']) if speaking_service else None
            if speaker_id:
                speaking_service.delete_speaker(speaker_id)
            practice.delete_student_data(account['student_id'])
            store.delete_account(account['id'])
        response = JSONResponse({'ok':True})
        response.delete_cookie(COOKIE,path='/')
        return response

    app.include_router(router)
    return store
