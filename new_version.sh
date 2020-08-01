if [ $# -eq 0 ]; then
  echo "Need a number as argument..."
  exit 1
else
  bot="bots/v$1.py"
  if [ -f "$bot" ]; then
    read -p "File $bot already exists. Overwrite? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]
    then
      echo "Not overwriting..."
      exit 1
    fi
      echo "Overwriting..."
      cat changelog.txt agent/base.py  > submission.py
      cp submission.py "$bot"
  fi
  echo "Creating $bot..."
  cat changelog.txt agent/base.py  > submission.py
  cp submission.py "$bot"
fi
