#ffmpeg -i jre.mp4 -vn -acodec copy jre.m4a
#ffmpeg -i jre.mp4 img%04d.png
#ffmpeg -i img%04d.png -i jre.m4a -c:v libx264 -r 25 -pix_fmt yuv420p -c:a copy -shortest out.mp4
ffmpeg -i img%04d.png -i jre.m4a -c:v libx264 -pix_fmt yuv420p -c:a copy -shortest out.mp4

