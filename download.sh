# download jieba dictionary
dict="dict.txt.zip","1jxdJOdPRy1HceAEVVr3cB8jlhXXxJ7PD"

files_arr=($dict)
for f in ${files_arr[@]}; do
    IFS=',' read filename fileid <<< "${f}"
    curl -L -o "${filename}" "https://drive.google.com/uc?export=download&id=${fileid}"
    echo "${filename}" ":" "${fileid} downloaded!"
    if [[ ${filename} = *"zip"* ]]; then
        unzip ${filename}
        rm ${filename}
    elif [[ ${filename} = *"gz"* ]]; then
        gunzip ${filename}
        rm ${filename}
    elif [[ ${filename} = *".tar.bz"* ]]; then
        tar jxvf ${filename}
        rm ${filename}
    elif [[ ${filename} = *".tar.gz"* ]]; then
        tar zxvf ${filename}
        rm ${filename}
    fi
done

# download corpus for MLE
cd corpus; ./download.sh; cd ..;

# download corpus for sentiment classifier
cd sentiment_analysis/corpus; ./download.sh; cd ../../

# soft link jieba dictionary
#cd sentiment_analysis; ln -s ../dict.txt ./

# download fasttext chinese model
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.zh.300.bin.gz
gunzip cc.zh.300.bin.gz
