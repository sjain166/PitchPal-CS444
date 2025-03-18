import json
import language_tool_python


input_file = "/Users/sidpro/Desktop/WorkPlace/UIUC/Spring-25/CS 444/PitchPal-CS444/data/Pitch-Sample/sample02_transcription.txt"
output_file = "/Users/sidpro/Desktop/WorkPlace/UIUC/Spring-25/CS 444/PitchPal-CS444/data/Pitch-Sample/sample02_analysis.json"

with open(input_file, 'r') as file:
    transcribed_text = file.read()
tool = language_tool_python.LanguageTool('en-US')
matches = tool.check(transcribed_text)

analysis_results = []
for match in matches:
    analysis_results.append({
        "message": match.message,
        "sentence": match.context,
        "suggestions": match.replacements,
        "offset": match.offset,
    })

with open(output_file, 'w') as file:
    json.dump(analysis_results, file, indent=4)

