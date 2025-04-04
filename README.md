# PitchPal-CS444
## Run Automation
```
source myenv/bin/activate
cd src
python automation.py ../data/Pitch-Sample/{.wav file-name}
```
`Tests will be accumulated in the "src/tests/results" folder`

### Activate Python "myenv":
`source myenv/bin/activate`

### Convert .m4a to .wav file
`ffmpeg -i input.m4a output.wav`

### Make a spectogram
`sox audio_file.wav -n spectrogram`