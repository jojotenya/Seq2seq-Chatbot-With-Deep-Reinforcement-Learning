source="source.zip","166qRYj3Q2d9KGj00wFlFk7gmY6MXDtpM"
source_xhj="source_xhj.zip","1LSSRlPOFk8msn0aF2EjTvTlHSOioIdeZ"
source_ptt="source_ptt.zip","1-dlQ93IWKcImJVZmrqo_RRF0KJERKKEv"
target="target.zip","19oY3Xx0WNjeIHDAfxY064eHPfnNoTJY6"
target_xhj="target_xhj.zip","1Jcy3Bi11p3b3hND1s5GOt7Mp3uiuPjlg"
target_ptt="target_ptt.zip","1XSkp2Zum1_IhXAcxleHxt49NmZfipDF7"

files_arr=($source $source_xhj $source_ptt $target $target_ptt $target_xhj)
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

mv source source_input
mv target target_input
mkdir pool
mv *_ptt pool
mv *_xhj pool
