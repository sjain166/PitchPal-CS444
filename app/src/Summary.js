import React from 'react';
import ReactMarkdown from 'react-markdown';

function Summary({audioDuration, summaryData}) {
  return (
    <div style={{ marginTop: '25px', fontSize: '17px' }}>
      <h2>Brief Summary</h2>
      {audioDuration === 0 ? (
        <p style={{ color: '#888' }}>Upload an audio file to see the summary.</p>
      ) : (
        <ReactMarkdown>{summaryData}</ReactMarkdown>
      )}
    </div>
  );
}

export default Summary;