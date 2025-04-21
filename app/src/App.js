import React, { useEffect, useRef, useState } from 'react';
import './App.css'; // For now, assume the styles are moved here

function App() {
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);
  const progressSliderRef = useRef(null);
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
    frequency: false
  });

  // Sample data
  const [emotionData] = useState([{ start_time: 0.0, end_time: 2.5 }, { start_time: 5.0, end_time: 7.0 }]);
  const [fillerData] = useState([{ word: "[UH]", timestamps: [{ start_time: 1.0, end_time: 1.5 }] }]);
  const [speedData] = useState([{ start_time: 2.5, end_time: 3.5, wpm: 80, status: "fast" }]);
  const [inappropriateData] = useState([{ word: "gonna", category: "unprofessional", start_time: 4.0, end_time: 4.2, confidence: 0.95 }]);
  const [volumeData] = useState({ loud: [], inaudible: [{ start_time: 6.0, end_time: 6.5 }] });
  const [frequencyData] = useState([{ word: "hi", count: 3, timestamps: [{ start_time: 3.0, end_time: 3.2 }, { start_time: 8.0, end_time: 8.3 }, { start_time: 10.0, end_time: 10.3 }] }]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateProgress = () => {
      const percent = (audio.currentTime / audio.duration) * 100;
      if (progressBarRef.current) progressBarRef.current.style.width = `${percent}%`;
    };

    audio.addEventListener('timeupdate', updateProgress);
    return () => audio.removeEventListener('timeupdate', updateProgress);
  }, []);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && audioRef.current) {
      audioRef.current.src = URL.createObjectURL(file);
      audioRef.current.load();

      audioRef.current.onloadedmetadata = () => {
        setAudioDuration(audioRef.current?.duration || 0);
        renderHighlights();
        renderTimestamps();
      };
    }
  };

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

  const toggleLayer = (layer) => {
    setLayers(prev => ({ ...prev, [layer]: !prev[layer] }));
  };

  const colorMap = {
    emotion: { backgroundColor: 'rgba(255,0,0,0.3)', borderColor: 'red' },
    filler: { backgroundColor: 'rgba(0,128,255,0.3)', borderColor: 'blue' },
    speed: { backgroundColor: 'rgba(0,255,128,0.3)', borderColor: 'green' },
    inappropriate: { backgroundColor: 'rgba(255,165,0,0.4)', borderColor: 'orange' },
    volume: { backgroundColor: 'rgba(128,0,255,0.3)', borderColor: 'purple' },
    frequency: { backgroundColor: 'rgba(255,255,0,0.4)', borderColor: '#999933' }
  };

  useEffect(() => {
    renderHighlights();
  }, [layers, audioDuration]);

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h2>Emotion-Based Audio Analysis</h2>
      <input type="file" ref={fileInputRef} accept="audio/*" onChange={handleFileChange} />
      <br /><br />
      <div style={{ marginTop: '20px' }}>
        <audio ref={audioRef} style={{ width: '100%', marginTop: '30px', marginBottom: '20px' }} controls />
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
      </div>
      <div style={{ display: 'flex' }}>
        <div style={{ width: 250 }}>
          <strong style={{ display: 'block', marginBottom: '10px' }}>Legend:</strong>
          {Object.entries({
            emotion: 'emotion detection',
            filler: 'filler words',
            speed: 'speaking speed',
            inappropriate: 'inappropriate words',
            volume: 'volume check',
            frequency: 'overused words'
          }).map(([key, label]) => (
            <label key={key} className="legend-item" style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={() => toggleLayer(key)}
                style={{ marginRight: '8px' }}
              />
              <span>{label}</span>
              <span
                className="legend-color"
                style={{
                  marginLeft: 'auto',
                  width: '20px',
                  height: '15px',
                  display: 'inline-block',
                  border: '1px solid',
                  backgroundColor: colorMap[key].backgroundColor,
                  borderColor: colorMap[key].borderColor
                }}
              ></span>
            </label>
          ))}
        </div>
        <div style={{ flex: 1, paddingLeft: '80px' }}>
          {audioDuration === 0 && (
            <div style={{ fontSize: '16px', color: '#888' }}>
              Upload an audio file to view analysis summaries here.
            </div>
          )}
          {audioDuration > 0 && (
            <>
              <div id="emotionSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Detected Emotions:</strong><br />
                {emotionData.map(({ start_time, end_time }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → emotion detected
                  </div>
                ))}
              </div>
              
              <div id="fillerSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Filler Words:</strong><br />
                {fillerData.map(({ word, timestamps }, idx) => (
                  <div key={idx}>
                    {timestamps.map((t, i) => (
                      <div key={i}>
                        from {t.start_time.toFixed(2)}s to {t.end_time.toFixed(2)}s → {word}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              
              <div id="speedSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Speaking Speed:</strong><br />
                {speedData.map(({ start_time, end_time, status }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → {status} speed
                  </div>
                ))}
              </div>
              
              <div id="inappropriateSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Inappropriate Words:</strong><br />
                {inappropriateData.map(({ start_time, end_time, category }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → {category} word spoken
                  </div>
                ))}
              </div>
              
              <div id="volumeSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Volume Issues:</strong><br />
                {volumeData.inaudible.map(({ start_time, end_time }, idx) => (
                  <div key={`inaudible-${idx}`}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → inaudible sound
                  </div>
                ))}
                {volumeData.loud.map(({ start_time, end_time }, idx) => (
                  <div key={`loud-${idx}`}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → loud sound
                  </div>
                ))}
              </div>
              
              <div id="frequencySummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Overused Words:</strong><br />
                {frequencyData.map(({ word, count, timestamps }, idx) => (
                  <div key={idx}>
                    {word} spoken {count} times at{' '}
                    {timestamps.map(({ start_time, end_time }) =>
                      `${start_time.toFixed(2)}s to ${end_time.toFixed(2)}s`
                    ).join(', ')}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;