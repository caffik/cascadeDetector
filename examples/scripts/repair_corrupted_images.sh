#!/usr/bin/env bash

# This script will repair or remove the corrupted images in the given directory

if ! command -v mogrify &> /dev/null
then
    echo "mogrify command could not be found"
    echo "Please install ImageMagick"
    exit
fi

if [ -z "$1" ]
then
    echo "Please provide the directory path"
    exit
fi

if [ ! -d "$1" ]
then
    echo "Directory not found"
    exit
fi

cd "$1" || exit

MOGRIFY="$(mogrify *.jpg 2>&1)"

while IFS= read -r line
do
  FILE_NAME=$(echo "$line" | grep -o '[0-9]*.jpg')
  if [[ $line == *"Corrupt JPEG"* ]]
  then
    MOGRIFY_FILE="$(mogrify -strip $FILE_NAME 2>&1)"
    if [ -z "$MOGRIFY_FILE" ]
    then
        echo "File $FILE_NAME repaired"
    else
        echo "File $FILE_NAME could not be repaired"
        rm $FILE_NAME
        echo "File $FILE_NAME removed"
    fi
  else
    echo "File $FILE_NAME could not be repaired"
    rm $FILE_NAME
    echo "File $FILE_NAME removed"
  fi
done <<< "$MOGRIFY"
