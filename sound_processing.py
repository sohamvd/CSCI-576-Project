import cv2
import librosa
import numpy as np
from scipy.io.wavfile import read as readWav
import json
from moviepy.editor import VideoFileClip

def audioFeatures(videoPath, audioPath):
    retDict = {}
    cap = cv2.VideoCapture(videoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    samplingRate, audioData = readWav(audioPath)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    audioFeatures = []
    vid = VideoFileClip(videoPath)
    for i, fr in enumerate(vid.iter_frames()):
        vidaudioStr = vid.audio.subclip((i / fps), ((i + 1) / fps))
        vidaudioStr = vidaudioStr.to_soundarray(fps=samplingRate)
        audioWindow = np.mean(vidaudioStr, axis=1)
        resAud = librosa.feature.mfcc(y=audioWindow, sr=samplingRate, n_mfcc=13)
        resMean = np.mean(resAud, axis=1)
        audioFeatures.append(resMean.tolist())

    cap.release()
    retDict["audio_features"] = audioFeatures
    retDict["fps"] = fps
    retDict["total_frames"] = totalFrames
    retDict["sampling_rate"] = samplingRate
    return retDict

def saveAudioFeaturesAsJson(audioFeatures, outputFile):
    with open(outputFile, 'w') as jsonFile:
        json.dump(audioFeatures, jsonFile, indent=4)


def preProcessAll():
    audioPath = './audiofiles/video'
    videoPath = './videofiles/video'
    outputPath = './featurefiles/video'
    for i in range(1, 21):
        audioFeaturesOutput = audioFeatures((videoPath+str(i)+'.mp4'), (audioPath+str(i)+'.wav'))
        saveAudioFeaturesAsJson(audioFeaturesOutput, (outputPath+str(i)+'.json'))
    
if __name__ == "__main__":
    preProcessAll()