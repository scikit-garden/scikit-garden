#!/bin/bash
# Almost copied verbatim from https://github.com/scikit-learn/scikit-learn/blob/master/build_tools/circle/push_doc.sh

if [ -z $CIRCLE_PROJECT_USERNAME ];
then USERNAME="skgardenci";
else USERNAME=$CIRCLE_PROJECT_USERNAME;
fi

MSG="Pushing the docs for revision for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

# Copy generated html by mkdocs to github pages
echo "Copying built files"
git clone -b master "git@github.com:scikit-garden/scikit-garden.github.io" deploy
cp -r ${HOME}/scikit-garden/site/* deploy

# Move into deployment directory
cd deploy

# Commit changes, allowing empty changes (when unchanged)
echo "Committing and pushing to Github"
echo "$USERNAME"
git config --global user.name $USERNAME
git config --global user.email "scikitgarden@gmail.com"
git config --global push.default matching
git add -A
git commit --allow-empty -m "$MSG"
git push

echo "$MSG"
