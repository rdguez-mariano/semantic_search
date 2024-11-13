# prevent docker from rebuild python dependencies at every little change we make to src/ folders

cp -r  /packages /tmp/packages
for folder in $(find /tmp/packages -mindepth 1 -maxdepth 1 -type d); do
    mkdir -p $folder/src/nqs
    touch $folder/src/nqs/__init__.py
    touch $folder/README.md
done
