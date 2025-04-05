import React, { useState } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.type === 'audio/wav') {
      setFile(selectedFile);
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
      {results.length > 0 && (
        <div>
          <h3>Analysis Results:</h3>
          <ul>
            {results.map((filename, index) => (
              <li key={index}>{filename}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
