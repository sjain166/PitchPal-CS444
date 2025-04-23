import React, { useEffect, useRef, useState } from 'react';
import './App.css';

function App() {
  const audioRef = useRef(null);
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

  const [emotionData, setEmotionData] = useState([]);
  const [fillerData, setFillerData] = useState([]);
  const [speedData, setSpeedData] = useState([]);
  const [inappropriateData, setInappropriateData] = useState([]);
  const [volumeData, setVolumeData] = useState({ loud: [], inaudible: [] });
  const [frequencyData, setFrequencyData] = useState([]);
  const [englishData, setEnglishData] = useState([]);

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
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [
          emotionRes,
          fillerRes,
          speedRes,
          inappropriateRes,
          volumeRes,
          frequencyRes,
          englishRes
        ] = await Promise.all([
          fetch('/analysis/emotion_analysis.json'),
          fetch('/analysis/filler_report.json'),
          fetch('/analysis/speech_rate_analysis.json'),
          fetch('/analysis/profanity_report.json'),
          fetch('/analysis/volume_report.json'),
          fetch('/analysis/word_frequency_report.json'),
          fetch('/analysis/sentence_structure_report.json')
        ]);

        const [emotionJson, fillerJson, speedJson, inappropriateJson, volumeJson, frequencyJson, englishJson] = await Promise.all([
          emotionRes.json(), fillerRes.json(), speedRes.json(), inappropriateRes.json(), volumeRes.json(), frequencyRes.json(), englishRes.json()
        ]);

        setEmotionData(emotionJson);
        setFillerData(fillerJson);
        setSpeedData(speedJson);
        setInappropriateData(inappropriateJson);
        setVolumeData(volumeJson);
        setFrequencyData(frequencyJson);
        setEnglishData(englishJson);

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
    frequency: { backgroundColor: 'rgba(255,255,0,0.4)', borderColor: '#999933' }
  };

  useEffect(() => {
    renderHighlights();
  }, [layers, audioDuration]);

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif' }}>
      <h1>PitchPal - Audio Analysis</h1>
      
      <div style={{ marginTop: '20px' }}>
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
                {emotionData.length === 0 ? (
                  <div>No emotion data available.</div>
                ) : (
                  emotionData.map(({ start_time, end_time, predicted_emotion }, idx) => (
                    <div key={idx}>
                      from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → "{predicted_emotion}" detected
                    </div>
                  ))
                )}
              </div>
              
              <div id="fillerSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Filler Words:</strong><br />
                {fillerData.length === 0 ? (
                  <div>No filler word data available.</div>
                ) : (
                  fillerData.map(({ word, timestamps }, idx) => (
                    <div key={idx}>
                      {timestamps.map((t, i) => (
                        <div key={i}>
                          from {t.start_time.toFixed(2)}s to {t.end_time.toFixed(2)}s → {word}
                        </div>
                      ))}
                    </div>
                  ))
                )}
              </div>
              
              <div id="speedSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Speaking Speed:</strong><br />
                {speedData.length === 0 ? (
                  <div>No speaking speed data available.</div>
                ) : (
                  speedData.map(({ start_time, end_time, status }, idx) => (
                    <div key={idx}>
                      from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → {status} speed
                    </div>
                  ))
                )}
              </div>
              
              <div id="inappropriateSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Inappropriate Words:</strong><br />
                {inappropriateData.length === 0 ? (
                  <div>No inappropriate word data available.</div>
                ) : (
                  inappropriateData.map(({ start_time, end_time, category, word }, idx) => (
                    <div key={idx}>
                      from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → {category} word spoken: {word}
                    </div>
                  ))
                )}
              </div>
              
              <div id="volumeSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Volume Issues:</strong><br />
                {volumeData.inaudible.length === 0 && volumeData.loud.length === 0 ? (
                  <div>No volume issues detected.</div>
                ) : (
                  <>
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
                  </>
                )}
              </div>
              
              <div id="frequencySummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Overused Words:</strong><br />
                {frequencyData.length === 0 ? (
                  <div>No overused word data available.</div>
                ) : (
                  frequencyData.map(({ word, count, timestamps }, idx) => (
                    <div key={idx}>
                      {word} spoken {count} times at{' '}
                      {timestamps.map(({ start_time, end_time }) =>
                        `${start_time.toFixed(2)}s to ${end_time.toFixed(2)}s`
                      ).join(', ')}
                    </div>
                  ))
                )}
              </div>

              <div id="englishSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
                <strong>Possible Grammatical Mistakes:</strong><br />
                {englishData.filter(item => item.corrected).length === 0 ? (
                  <div>No grammatical errors.</div>
                ) : (
                  <ul style={{ paddingLeft: '20px' }}>
                    {englishData
                      .filter(item => item.corrected)
                      .map(({ sentence, corrected }, idx) => (
                        <li key={`grammar-${idx}`}>
                          <strong>Original:</strong> {sentence}<br />
                          <strong>Suggested:</strong> {corrected}
                        </li>
                      ))}
                  </ul>
                )}
                <br />
                <strong>Irrelevant Sentences:</strong><br />
                {englishData.filter(item => !item.corrected).length === 0 ? (
                  <div>No irrelevant sentence used.</div>
                ) : (
                  <ul style={{ paddingLeft: '20px' }}>
                    {englishData
                      .filter(item => !item.corrected)
                      .map(({ sentence }, idx) => (
                        <li key={`irrelevant-${idx}`}>{sentence}</li>
                      ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;