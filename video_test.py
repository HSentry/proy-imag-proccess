from moviepy.editor import VideoFileClip
from IPython.display import HTML
from final_function import lane_finding_pipeline



white_output = '.\proy-imag-proccess'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#clip1 = VideoFileClip("test_videos/jaipurHighway.mp4").subclip(50,60)
clip1 = VideoFileClip("IMG_0264.mp4")
white_clip = clip1.fl_image(lane_finding_pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False, codec="libx264")

