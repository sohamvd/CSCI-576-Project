import numpy as np
import json
import sound_processing as sp
import subprocess

def play_video_from_timestamp(timestamp, video_path):
    timestamp_formatted = f"--start-time={timestamp}"
    command = ['vlc', '--play-and-stop', timestamp_formatted, video_path]
    subprocess.run(command)

def load_video_features_from_json(input_file):
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)
        video_features = data
        if not isinstance(video_features['audio_features'], list):
            print(f"Warning: Invalid video features data loaded from {input_file}. Returning empty list.")
            return []
    return video_features


def audio_match(query_features_path, original_features_path):
    query_audio_features = query_features_path['audio_features']
    original_audio_features = load_video_features_from_json(original_features_path)['audio_features']

    query_mfcc = np.stack(query_audio_features)
    query_len = len(query_mfcc)
    window_len = min(query_len, 600)
    query_mfcc_window = query_mfcc[:window_len].flatten()

    original_mfcc = np.stack(original_audio_features)
    original_len = len(original_mfcc)
    output = []

    for start in range(original_len - window_len):
        cosineVal = np.dot(query_mfcc_window, original_mfcc[start:start+window_len].flatten()) / (np.linalg.norm(query_mfcc_window)*np.linalg.norm(original_mfcc[start:start+window_len].flatten()))
        result = cosineVal
        frameTime = start / query_features_path['fps']
        output.append({
            'result': result,
            'video': original_features_path,
            'frame_no': start,
            'frame_time': frameTime
        })
    output.sort(reverse=True, key=lambda i: i['result'])

    return output

def search(queryVideoPath, queryAudioPath):
    queryAudioFeatures = sp.audioFeatures(queryVideoPath, queryAudioPath)
    allOutputs = []
    featurePath = './featurefiles/video'
    for i in range(1, 21):
        allOutputs+=(audio_match(queryAudioFeatures, (featurePath+str(i)+'.json')))
    
    allOutputs.sort(reverse=True, key=lambda i: i['result'])
    print(allOutputs[:1])
    return allOutputs[0]

allOutputs = search('./queryVid/video5_1_modified.mp4', './queryAud/video5_1_modified.wav')
path = allOutputs['video']
start_index = path.rfind('/') + 1
end_index = path.rfind('.')
video_name = path[start_index:end_index]
print(video_name)
play_video_from_timestamp(allOutputs['frame_time'], './videofiles/'+video_name+'.mp4')

# vidpath = './queryVid/video'
# audpath = './queryAud/video'
# for i in range(1, 11):
#     search(vidpath+str(i)+'_1_modified.mp4', audpath+str(i)+'_1_modified.wav')