import { useState, useRef, useCallback } from 'react';
import api from '../utils/api';

export default function useEmotionDetection() {
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [allScores, setAllScores] = useState({});
  const [emotionDisplay, setEmotionDisplay] = useState(null);
  const [faceFound, setFaceFound] = useState(false);
  const [bbox, setBbox] = useState(null);
  const [frameSize, setFrameSize] = useState(null);
  const [detecting, setDetecting] = useState(false);
  const [history, setHistory] = useState([]);
  const lastDetect = useRef(0);

  const detect = useCallback(async (base64Frame) => {
    if (!base64Frame) return null;
    const now = Date.now();
    if (now - lastDetect.current < 1500) return null; // throttle 1.5s
    lastDetect.current = now;
    setDetecting(true);

    try {
      const result = await api.detectEmotion(base64Frame);
      setEmotion(result.emotion);
      setConfidence(result.confidence);
      setAllScores(result.all_scores || {});
      setEmotionDisplay({ name: result.display_name, meaning: result.display_meaning, emoji: result.display_emoji });
      setFaceFound(result.face_found);
      setBbox(result.bbox || null);
      setFrameSize(result.frame_size || null);

      if (result.face_found) {
        setHistory(prev => [...prev.slice(-49), { emotion: result.emotion, confidence: result.confidence, time: Date.now() }]);
      }
      setDetecting(false);
      return result;
    } catch (e) {
      setDetecting(false);
      return null;
    }
  }, []);

  return { emotion, confidence, allScores, emotionDisplay, faceFound, bbox, frameSize, detecting, history, detect };
}
