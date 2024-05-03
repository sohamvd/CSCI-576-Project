import vlc

# Path to the audio file
audio_path = "path_to_audio_file.wav"

# Create a VLC media player instance
player = vlc.MediaPlayer()

# Load the audio file
media = vlc.Media(audio_path)
player.set_media(media)

# Play the audio
player.play()

# Wait for the audio to finish
while player.is_playing():
    pass

# Release the player
player.release()
