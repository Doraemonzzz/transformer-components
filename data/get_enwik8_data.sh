echo "- Downloading enwik8 (Character)"
if [[ ! -d './.data/enwik8' ]]; then
    mkdir -p ./.data/enwik8
fi

cd ./.data/enwik8
wget --continue http://mattmahoney.net/dc/enwik8.zip