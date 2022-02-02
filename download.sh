# Download batches codes
wget -c -O cup_annotations_test "https://storage.googleapis.com/objectron/v1/index/cup_annotations_test"
FILE="./cup_annotations_test"

while read p; do
    # Generate unique names
    filename=$(echo $p | tr /- _)
    
    # Download Files
    wget -c -O "${filename}_video.mov" "https://storage.googleapis.com/objectron/videos/$p/video.MOV"
    wget -c -O "${filename}_metadata.pbdata" "https://storage.googleapis.com/objectron/videos/$p/geometry.pbdata"
    wget -c -O "${filename}_annotation.pbdata" "https://storage.googleapis.com/objectron/annotations/$p.pbdata"
done < $FILE
