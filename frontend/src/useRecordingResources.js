import {useEffect, useMemo} from 'react';

export function discardRecording(recorder) {
  if (!recorder) return;
  recorder.onstop = null;
  recorder.ondataavailable = null;
  if (recorder.state === 'recording') {
    recorder.stop();
  }
  recorder.stream?.getTracks().forEach(track => track.stop());
}

export function recordingFileName(blob, prefix = 'recording') {
  const type = blob?.type || '';
  const extension = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
  return `${prefix}.${extension}`;
}

export function useRecordingResources(recorderRef, blob) {
  const blobUrl = useMemo(() => blob ? URL.createObjectURL(blob) : null, [blob]);
  useEffect(() => () => { if (blobUrl) URL.revokeObjectURL(blobUrl); }, [blobUrl]);
  useEffect(() => () => discardRecording(recorderRef.current), [recorderRef]);
  return blobUrl;
}
