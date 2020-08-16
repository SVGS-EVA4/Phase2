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
        url:'https://7odoljgr3c.execute-api.ap-south-1.amazonaws.com/dev/face_alignment',
        data: formData,
        processData: false,
        
        contentType: false,
        
   
        mimeType:'multipart/form-data',


    })
    .done(function(response){
        console.log('hello')
        
        document.getElementById('result').src = 'data:image/jpeg;base64,'+(JSON.parse(response)).ImageBytes;
    })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#btnUpload').click(uploadAndAlignFace);
















