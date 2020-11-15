cd ../input
wget https://facultyweb.cs.wwu.edu/~wehrwes/courses/csci497p_20f/p3/tentacle.zip
unzip tentacle.zip
cd ../data

cd ../test_materials
wget https://facultyweb.cs.wwu.edu/~wehrwes/courses/csci497p_20f/p3/fabrics_normalized.npy
cd ../data

#wget --no-check-certificate http://vision.seas.harvard.edu/qsfs/PSData.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Adirondack-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Backpack-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Bicycle1-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Cable-perfect.zip



#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Classroom1-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Couch-perfect.zip
wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Flowers-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Jadeplant-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Mask-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Motorcycle-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Piano-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Playroom-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Recycle-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Shelves-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Shopvac-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Sticks-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Storage-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword1-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Sword2-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Umbrella-perfect.zip
#wget --no-check-certificate http://vision.middlebury.edu/stereo/data/scenes2014/zip/Vintage-perfect.zip

find . -name "*.zip" -exec unzip {} \;
