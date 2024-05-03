import vlc

# Path to the audio file
audio_path = "./audiofiles/video9.wav"

# Create a VLC media player instance
player = vlc.MediaPlayer('./audiofiles/video9.wav')

# Play the audio
player.play()

# Wait for the audio to finish
while player.is_playing():
    pass

# Release the player
player.release()
