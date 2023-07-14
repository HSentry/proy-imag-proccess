from moviepy.editor import VideoFileClip
from IPython.display import HTML
from final_function import lane_finding_pipeline



white_output = './output_test2.mp4'
clip1 = VideoFileClip("Data_Set1.mp4")
white_clip = clip1.fl_image(lane_finding_pipeline) 
white_clip.write_videofile(white_output, audio=False, codec="libx264")

