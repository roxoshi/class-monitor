const submitButton = document.querySelector('#fnameButton');

submitButton.addEventListener('click', function(e){
	e.preventDefault()

	let tweetText = document.querySelector('#fnameText');
	console.log('tweetText var is:',tweetText);
	var tweetObj = JSON.stringify({
		tweet: tweetText.value,
	})
	console.log('body is: ', tweetObj);
	fetch('http://127.0.0.1:5000/', {
		method: 'POST',
		body: tweetObj,
		headers: {
			'Content-type': 'application/json; charset=UTF-8',
		}

	})
	.then(function(response){
		responseClone = response.clone();
		return response.json()})
	.then(function(data){
		console.log(data);
		let flagger = data.prediction;
		console.log('flagger is: ',flagger);
		let tweetOutput = document.querySelector('.predictOutput');
		if (flagger === "1") {
			tweetOutput.textContent = "This is a hate tweet";
		} else {
			tweetOutput.textContent = "This tweet OK!";
		}
		 
	})
	.catch(error => console.error('Error:', error, 'response: ',responseClone.text()));
}
)

