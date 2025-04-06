import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [showProfanity, setShowProfanity] = useState(false);
  const [profanityData, setProfanityData] = useState<any[]>([]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'audio/wav') {
      setFile(selectedFile);
      setAudioUrl(URL.createObjectURL(selectedFile)); // ✅ generate audio URL
      setMessage('');
    } else {
      setMessage('Please upload a .wav audio file');
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setMessage('Please upload a file first');
      return;
    }

    const formData = new FormData();
    formData.append('audio', file);
    setLoading(true);
    setMessage('Analyzing...');

    try {
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setResults(data.results);
        setMessage(data.message);
        console.log(data);
        const profanityRes = await fetch('../../../src/tests/results/profanity_report.json');
        console.log(profanityRes)
        const profanityJson = await profanityRes.json();
        console.log(profanityJson)
        setProfanityData(profanityJson);
      } else {
        setMessage(data.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Error:', error);
      setMessage('Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>PitchPal Dashboard</h1>
      <input type="file" accept=".wav" onChange={handleFileChange} />
      <button onClick={handleAnalyze} disabled={loading}>
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
      {message && <p>{message}</p>}
      {audioUrl && results.length > 0 && (
        <div style={{ marginTop: '40px', padding: '20px', borderTop: '2px solid #ddd', width: '90%', marginLeft: 'auto', marginRight: 'auto' }}>
          <audio ref={audioRef} src={audioUrl} controls style={{ width: '100%' }} />
          <div style={{ display: 'flex', marginTop: '20px' }}>
            <div style={{ flex: 1 }}>
              <label>
                <input type="radio" checked={showProfanity} onChange={() => setShowProfanity(!showProfanity)} />
                Profanity Check
              </label>
            </div>
            <textarea
              style={{ flex: 2, height: '100px', marginLeft: '20px' }}
              placeholder="Comments"
              readOnly
              value={showProfanity ? '🔴 Profanity detected. Feedback will appear here.' : ''}
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
