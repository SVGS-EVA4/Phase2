function uploadAndSwapFace(){
    var fileInput = document.getElementById('fileUpload').files;
    if(!fileInput.length){
        return alert('Please choose image to swap face.')
    }
    var file = fileInput[0];
    var filename = file.name;
    var formData = new FormData();
    formData.append(filename,file);
    console.log(filename);
	
	var fileInput1 = document.getElementById('imgFileUpload').files;
    if(!fileInput1.length){
        return alert('Please choose image on which to swap face.')
    }
    var file1 = fileInput1[0];
    var filename1 = file1.name;
    formData.append(filename1,file1);
    console.log(filename1);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url:'https://xxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/face_swap',
        data: formData,
        processData: false,
        
        contentType: false,
        
   
        mimeType:'multipart/form-data',


    })
    .done(function(response){
        console.log('hello')
        console.log(response);
        if (((JSON.parse(response)).Status) =='IncorrectInput'){
            document.getElementById('result').src = '/';
            document.getElementById('error').textContent = 'Face missing.';
        }
        else{
            // var spantag = document.getElementById('spantag');
            // var imgtag = document.createElement("IMG");
            // spantag.appendChild(imgtag)
            // <img id='result' src='/' alt='result' width="200" />
            document.getElementById('result_msg').textContent = 'Swapped Image: '
            document.getElementById('result_msg').style.display = 'inherit'
            document.getElementById('error').textContent = '';
            document.getElementById('result').src = 'data:image/jpeg;base64,'+(JSON.parse(response)).ImageBytes;
            document.getElementById('result').style.display = 'inherit'

        }
    })
    .fail(function(){
        alert('Error during swapping');
    });
};

$('#btnUpload').click(uploadAndSwapFace);