function uploadAndClassifyImageMobilenet(){
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
        url:'https://l485dsw7nc.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        
        contentType: false,
        
   
        mimeType:'multipart/form-data',


    })
    .done(function(response){
        console.log(response);
        document.getElementById('result1').textContent = response;
    })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#btnUploadMobilenet').click(uploadAndClassifyImageMobilenet);

function uploadAndClassifyImageFourClass(){
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
        url:'https://865fgqaq94.execute-api.ap-south-1.amazonaws.com/dev/classification',
        
        data: formData,
        processData: false,
        
        contentType: false,
        
   
        mimeType:'multipart/form-data',


    })
    .done(function(response){
        console.log(response);
        document.getElementById('result2').textContent = response;
    })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#btnUploadFourClass').click(uploadAndClassifyImageFourClass);
















