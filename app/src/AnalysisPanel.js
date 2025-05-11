import React from 'react';

function AnalysisPanel({
  layers,
  toggleLayer,
  colorMap,
  audioDuration,
  emotionData,
  fillerData,
  speedData,
  inappropriateData,
  volumeData,
  frequencyData,
  englishData,
  v_nervous,
  v_eye,
  v_bg
}) {
  return (
    <div style={{ display: 'flex' }}>
      {/* ---------- Left-hand legend ---------- */}
      <div style={{ width: 250 }}>
        <strong style={{ display: 'block', marginBottom: '10px' }}>Audio Legend:</strong>
        {Object.entries({
          emotion: 'emotion detection',
          filler: 'filler words',
          speed: 'speaking speed',
          inappropriate: 'inappropriate words',
          volume: 'volume check',
          frequency: 'overused words',
        }).map(([key, label]) => (
          <label
            key={key}
            className="legend-item"
            style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}
          >
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
                borderColor: colorMap[key].borderColor,
              }}
            />
          </label>
        ))}
        <hr style={{ height: 1, backgroundColor: '#e2e8f0', margin: '24px 0' }} />
        <strong style={{ display: 'block', marginBottom: '10px' }}>Video Legend:</strong>
        {Object.entries({
          v_nervous: 'Nervousness',
          v_eye: 'Poor Eye Contact',
          v_bg: 'Backgroud Noise'
        }).map(([key, label]) => (
          <label
            key={key}
            className="legend-item"
            style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}
          >
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
                borderColor: colorMap[key].borderColor,
              }}
            />
          </label>
        ))}
      </div>
      

      {/* ---------- Right-hand summaries ---------- */}
      <div style={{ flex: 1, paddingLeft: '80px' }}>
        {audioDuration === 0 && (
          <div style={{ fontSize: '16px', color: '#888' }}>
            Upload an audio file to view analysis summaries here.
          </div>
        )}

        {audioDuration > 0 && (
          <>
            {/* Emotion summary */}
            <div id="emotionSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Detected Emotions:</strong>
              <br />
              {emotionData.length === 0 ? (
                <div>No emotion data available.</div>
              ) : (
                emotionData.map(({ start_time, end_time, predicted_emotion }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → "
                    {predicted_emotion}" detected
                  </div>
                ))
              )}
            </div>

            {/* Filler summary */}
            <div id="fillerSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Filler Words:</strong>
              <br />
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

            {/* Speed summary */}
            <div id="speedSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Speaking Speed:</strong>
              <br />
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

            {/* Inappropriate words summary */}
            <div id="inappropriateSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Inappropriate Words:</strong>
              <br />
              {inappropriateData.length === 0 ? (
                <div>No inappropriate word data available.</div>
              ) : (
                inappropriateData.map(({ start_time, end_time, category, word }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s → {category} word
                    spoken: {word}
                  </div>
                ))
              )}
            </div>

            {/* Volume summary */}
            <div id="volumeSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Volume Issues:</strong>
              <br />
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

            {/* Frequency summary */}
            <div id="frequencySummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Overused Words:</strong>
              <br />
              {frequencyData.length === 0 ? (
                <div>No overused word data available.</div>
              ) : (
                frequencyData.map(({ word, count, timestamps }, idx) => (
                  <div key={idx}>
                    {word} spoken {count} times at{' '}
                    {timestamps
                      .map(({ start_time, end_time }) => `${start_time.toFixed(2)}s to ${end_time.toFixed(2)}s`)
                      .join(', ')}
                  </div>
                ))
              )}
            </div>

            {/* English summary */}
            <div id="englishSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Possible Grammatical Mistakes:</strong>
              <br />
              {englishData.filter((item) => item.corrected).length === 0 ? (
                <div>No grammatical errors.</div>
              ) : (
                <ul style={{ paddingLeft: '20px' }}>
                  {englishData
                    .filter((item) => item.corrected)
                    .map(({ sentence, corrected }, idx) => (
                      <li key={`grammar-${idx}`}>
                        <strong>Original:</strong> {sentence}
                        <br />
                        <strong>Suggested:</strong> {corrected}
                      </li>
                    ))}
                </ul>
              )}
              <br />
              <strong>Irrelevant Sentences:</strong>
              <br />
              {englishData.filter((item) => !item.corrected).length === 0 ? (
                <div>No irrelevant sentence used.</div>
              ) : (
                <ul style={{ paddingLeft: '20px' }}>
                  {englishData
                    .filter((item) => !item.corrected)
                    .map(({ sentence }, idx) => (
                      <li key={`irrelevant-${idx}`}>{sentence}</li>
                    ))}
                </ul>
              )}
            </div>
          </>
        )}
      </div>
      <div style={{ flex: 1, paddingLeft: '80px' }}>
        {audioDuration === 0 && (
          <div style={{ fontSize: '16px', color: '#888' }}>
            Upload an video file to view analysis summaries here.
          </div>
        )}

        {audioDuration > 0 && (
          <>
            {/* Nervousness summary */}
            <div id="emotionSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Nervousness:</strong>
              <br />
              {v_nervous.length === 0 ? (
                <div>No nervous data available.</div>
              ) : (
                v_nervous.map(({ start_time, end_time }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s
                  </div>
                ))
              )}
            </div>
            
            {/* Eye Contact summary */}
            <div id="emotionSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Poor Eye Contact:</strong>
              <br />
              {v_eye.length === 0 ? (
                <div>No nervous data available.</div>
              ) : (
                v_eye.map(({ start_time, end_time }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s
                  </div>
                ))
              )}
            </div>

            {/* Background Noise summary */}
            <div id="emotionSummary" style={{ marginTop: '15px', fontSize: '16px' }}>
              <strong>Background Noise:</strong>
              <br />
              {v_bg.length === 0 ? (
                <div>No nervous data available.</div>
              ) : (
                v_bg.map(({ start_time, end_time }, idx) => (
                  <div key={idx}>
                    from {start_time.toFixed(2)}s to {end_time.toFixed(2)}s
                  </div>
                ))
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default AnalysisPanel;