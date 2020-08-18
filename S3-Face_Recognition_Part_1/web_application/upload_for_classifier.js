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
        let file_name = (JSON.parse(response)).File;
        let predictions = (JSON.parse(response))['Predicted Class'];
        // document.getElementById('result1').textContent ='Classify Image: ' ;
        document.getElementById('result1a').textContent =predictions + '!';
        document.getElementById('result1b').textContent = 'Pretrained Resnet Model says it as '  ;  
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
        let file_name1 = (JSON.parse(response)).File;
        let predictions1 = (JSON.parse(response))['Predicted Class'];
        // document.getElementById('result2').textContent ='Classify Birds and Drones: ' ;
        document.getElementById('result2a').textContent =predictions1+'!' ;
        document.getElementById('result2b').textContent = 'Custom pretrained Mobilenet says it as  ' ; 
    })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#btnUploadFourClass').click(uploadAndClassifyImageFourClass);

