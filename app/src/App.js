import React, { useEffect, useRef, useState } from 'react';
import './App.css';
import AnalysisPanel from './AnalysisPanel';
import Summary from './Summary';

function App() {
  const audioRef = useRef(null);
  const videoRef = useRef(null); 
  const highlightLayerRef = useRef(null);
  const progressBarRef = useRef(null);
  const timestampLabelsRef = useRef(null);

  const [audioDuration, setAudioDuration] = useState(0);
  const [layers, setLayers] = useState({
    emotion: false,
    filler: false,
    speed: false,
    inappropriate: false,
    volume: false,
    frequency: false,
    v_nervous: false,
    v_eye: false,
    v_bg: false,
  });
  const [view, setView] = useState('summary');

  const [emotionData, setEmotionData] = useState([]);
  const [fillerData, setFillerData] = useState([]);
  const [speedData, setSpeedData] = useState([]);
  const [inappropriateData, setInappropriateData] = useState([]);
  const [volumeData, setVolumeData] = useState({ loud: [], inaudible: [] });
  const [frequencyData, setFrequencyData] = useState([]);
  const [englishData, setEnglishData] = useState([]);
  const [summary, setSummary] = useState("");
  const [raw_transcribed_text, setRawTranscribedText] = useState("");
  const [video_nervous, setVideoNervous] = useState([]);
  const [video_eye, setVideoEye] = useState([]);
  const [video_bg_noise, setVideoBgNoise] = useState([]);

  useEffect(() => {
    const audio = audioRef.current;
    const video = videoRef.current;
    if (!audio) return;

    const updateProgress = () => {
      const percent = (audio.currentTime / audio.duration) * 100;
      if (progressBarRef.current) progressBarRef.current.style.width = `${percent}%`;
      if (video) video.currentTime = audio.currentTime;
    };

    const playVideo  = () => video && video.play();
    const pauseVideo = () => video && video.pause();

    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('play',  playVideo);
    audio.addEventListener('pause', pauseVideo);
    // return () => audio.removeEventListener('timeupdate', updateProgress);
    return () => {
      audio.removeEventListener('timeupdate', updateProgress);
      audio.removeEventListener('play',  playVideo);
      audio.removeEventListener('pause', pauseVideo);
    };
  }, []);

  const renderHighlights = () => {
    const container = highlightLayerRef.current;
    if (!container) return;
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    const addHighlight = (start, end, className) => {
      const div = document.createElement('div');
      div.className = `highlight-layer-segment ${className}`;
      div.style.left = `${(start / audioDuration) * 100}%`;
      div.style.width = `${((end - start) / audioDuration) * 100}%`;
      container.appendChild(div);
    };

    if (layers['emotion']) {
      emotionData.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'emotion-highlight')
      );
    }

    if (layers['filler']) {
      fillerData.forEach(({ timestamps }) => {
        timestamps.forEach(({ start_time, end_time }) =>
          addHighlight(start_time, end_time, 'filler-highlight')
        );
      });
    }

    if (layers['speed']) {
      speedData.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'speed-highlight')
      );
    }

    if (layers['inappropriate']) {
      inappropriateData.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'inappropriate-highlight')
      );
    }

    if (layers['volume']) {
      volumeData.inaudible.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'volume-highlight')
      );
      volumeData.loud.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'volume-highlight')
      );
    }

    if (layers['frequency']) {
      frequencyData.forEach(({ timestamps }) => {
        timestamps.forEach(({ start_time, end_time }) =>
          addHighlight(start_time, end_time, 'frequency-highlight')
        );
      });
    }

    if(layers['v_nervous']) {
      video_nervous.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'v_nervous-highlight')
      );
    }

    if(layers['v_eye']) {
      video_eye.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'v_eye-highlight')
      );
    }

    if(layers['v_bg']) {
      video_bg_noise.forEach(({ start_time, end_time }) =>
        addHighlight(start_time, end_time, 'v_bg-highlight')
      );
    }
  };

  const renderTimestamps = () => {
    const container = timestampLabelsRef.current;
    const audio = audioRef.current;
    if (!container || !audio || isNaN(audio.duration)) return;

    const duration = audio.duration;
    container.innerHTML = '';
    const numMarkers = 10;
    for (let i = 0; i <= numMarkers; i++) {
      const time = (i / numMarkers) * duration;
      const label = document.createElement('span');
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60);
      label.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
      container.appendChild(label);
    }
  };
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/analysis/combined.json');
        const data = await res.json();
        setEmotionData(data.emotion_analysis);
        setFillerData(data.filler_report);
        setSpeedData(data.speech_rate_analysis);
        setInappropriateData(data.profanity_report);
        setVolumeData(data.volume_report);
        setFrequencyData(data.word_frequency_report);
        setEnglishData(data.sentence_structure_report);
        setRawTranscribedText(data.raw_transcribed_text)
        setSummary(data.feedback)
        setVideoNervous(data.nervous_timeline);
        setVideoEye(data.eye_discontact_timeline);
        setVideoBgNoise(data.background_noise);
      } catch (err) {
        console.error('Failed to load analysis data:', err);
      }
    };

    fetchData();
  }, []);

  const toggleLayer = (layer) => {
    setLayers(prev => ({ ...prev, [layer]: !prev[layer] }));
  };

  const colorMap = {
    emotion: { backgroundColor: 'rgba(255,0,0,0.3)', borderColor: 'red' },
    filler: { backgroundColor: 'rgba(0,128,255,0.3)', borderColor: 'blue' },
    speed: { backgroundColor: 'rgba(0,255,128,0.3)', borderColor: 'green' },
    inappropriate: { backgroundColor: 'rgba(255,165,0,0.4)', borderColor: 'orange' },
    volume: { backgroundColor: 'rgba(128,0,255,0.3)', borderColor: 'purple' },
    frequency: { backgroundColor: 'rgba(255,255,0,0.4)', borderColor: '#999933' },
    v_nervous: { backgroundColor: 'rgba(255, 0, 255, 0.3)', borderColor: 'purple' },
    v_eye: { backgroundColor: 'rgba(0, 255, 255, 0.3)', borderColor: 'blue' },
    v_bg: { backgroundColor: 'rgba(128, 128, 0, 0.3)', borderColor: 'yellow' }
  };

  useEffect(() => {
    renderHighlights();
  }, [layers, audioDuration]);

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>PitchPal - Elevator Pitch Analysis</h1>
      <div style={{ marginTop: '10px' }}>
        <button onClick={() => setView('summary')} disabled={view === 'summary'}>
          Brief Summary
        </button>
        <button
          onClick={() => setView('analysis')}
          disabled={view === 'analysis'}
          style={{ marginLeft: '8px' }}
        >
          Analysis
        </button>
      </div>
      
      <div style={{ marginTop: '20px' }}>
        <video
          ref={videoRef}
          src="/analysis/video.mov"
          muted
          style={{ width: '100%', maxHeight: 400, marginTop: 10 }}
        />

        <audio ref={audioRef} style={{ width: '100%', marginTop: '30px', marginBottom: '20px' }} controls onLoadedMetadata={() => {
          if (audioRef.current) {
            setAudioDuration(audioRef.current.duration);
            renderHighlights();
            renderTimestamps();
          }
        }}>
          <source src="/analysis/audio.wav" type="audio/wav" />
          Your browser does not support the audio element.
        </audio>
        <div
          className="slider-container"
          style={{
            position: 'relative',
            width: '100%',
            height: '40px',
            background: 'repeating-linear-gradient(to right, #e0e0e0, #e0e0e0 2px, #ffffff 2px, #ffffff 4px)',
            marginTop: '10px',
            cursor: 'pointer'
          }}
        >
          <div ref={highlightLayerRef} className="highlight-layer"></div>
          <div ref={progressBarRef} style={{ position: 'absolute', height: '5px', background: 'black', top: 0, left: 0, width: '0%', zIndex: 5, pointerEvents: 'none' }}></div>
        </div>
        <div
          className="timestamp-labels"
          ref={timestampLabelsRef}
          style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px', color: '#555', marginTop: '5px' }}
        ></div>
        <hr style={{ height: 1, backgroundColor: '#e2e8f0', margin: '24px 0' }} />
        <h2>Transcribed Text</h2>
        <text>{raw_transcribed_text}</text>
        <hr style={{ height: 1, backgroundColor: '#e2e8f0', margin: '24px 0' }} />
      </div>
      {view === 'analysis' ? (
        <>
          <br/>
          <AnalysisPanel
            layers={layers}
            toggleLayer={toggleLayer}
            colorMap={colorMap}
            audioDuration={audioDuration}
            emotionData={emotionData}
            fillerData={fillerData}
            speedData={speedData}
            inappropriateData={inappropriateData}
            volumeData={volumeData}
            frequencyData={frequencyData}
            englishData={englishData}
            v_nervous={video_nervous}
            v_eye={video_eye}
            v_bg={video_bg_noise}
          />
        </>
      ) : (
        <Summary
          audioDuration={audioDuration}
          summaryData={summary}
        />
      )}
    </div>
  );
};

export default App;