function uploadAndAlignFace(){
    
    

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'GET',
        url:' https://zedptp9pc4.execute-api.ap-south-1.amazonaws.com/dev/gan',
        
        processData: false,
        
        contentType: false,
        
   
        mimeType:'multipart/form-data',


    })
    .done(function(response){
        console.log('hello')
        console.log((JSON.parse(response)).data.ImageBytes);
        if (((JSON.parse(response)).Status) =='IncorrectInput'){
            document.getElementById('result').src = '/';
            document.getElementById('error').textContent = 'Error, Retry';
        }
        else{
            // var spantag = document.getElementById('spantag');
            // var imgtag = document.createElement("IMG");
            // spantag.appendChild(imgtag)
            // <img id='result' src='/' alt='result' width="200" />
            document.getElementById('error').textContent = '';
            document.getElementById('result').src = 'data:image/jpeg;base64,'+(JSON.parse(response)).data;
            document.getElementById('result').style.display = 'inherit'
        }
    })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#btnUpload').click(uploadAndAlignFace);