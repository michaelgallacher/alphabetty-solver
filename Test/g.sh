if [[ $2 ]]; then
	python3 findletters.py -i $1 -f $2
else
	python3 findletters.py -i $1 -d 1
fi
