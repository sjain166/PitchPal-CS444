import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

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
          {/* <h3 style={{ textAlign: 'center' }}>🎧 RESULTS </h3> */}
          {/* <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '20px', marginBottom: '20px' }}> */}
            {/* <button onClick={() => {
              if (audioRef.current) audioRef.current.currentTime -= 5;
            }} style={{ fontSize: '24px' }}>⏪</button>

            <button onClick={() => audioRef.current?.play()} style={{ fontSize: '36px', borderRadius: '50%', width: '60px', height: '60px' }}>▶️</button>

            <button onClick={() => {
              if (audioRef.current) audioRef.current.currentTime += 5;
            }} style={{ fontSize: '24px' }}>⏩</button> */}
          {/* </div> */}

          <audio ref={audioRef} src={audioUrl} controls style={{ width: '100%' }} />
        </div>
      )}
    </div>
  );
}

export default App;
