if [[ $1 ]]; then
	python3 findletters.py -i $1 -d -1
else
	echo 'oops: filename is required'
fi
