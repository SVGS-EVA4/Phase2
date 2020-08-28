function uploadAndAlignFace(){
    var fileInput = document.getElementById('fileUpload').files;
    if(!fileInput.length){
        return alert('Please chose file')
    }
    var file = fileInput[0];
    var filename = file.name;
    var formData = new FormData();
    formData.append(filename,file);
    console.log(filename);

    $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url:'https://xxxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/face_alignment',
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
            document.getElementById('error').textContent = 'No faces detected';
        }
        else{
            // var spantag = document.getElementById('spantag');
            // var imgtag = document.createElement("IMG");
            // spantag.appendChild(imgtag)
            // <img id='result' src='/' alt='result' width="200" />
            document.getElementById('error').textContent = '';
            document.getElementById('result').src = 'data:image/jpeg;base64,'+(JSON.parse(response)).ImageBytes;
            document.getElementById('result').style.display = 'inherit'
        }
    })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#btnUpload').click(uploadAndAlignFace);