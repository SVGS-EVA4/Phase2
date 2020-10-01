function uploadAndRecogniseFace(){
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
        url:'https://xxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/face_recognition',
        data: formData,
        processData: false,
        
        contentType: false,
        
   
        mimeType:'multipart/form-data',


    })
    .done(function(response){
        console.log(response);

        let state =  (JSON.parse(response)).Status;
        if (state == '0'){
            alert('Please input correct image.');
        }
        else{
            let file_name = (JSON.parse(response)).File;
            let predictions = (JSON.parse(response))['Predicted_Class'];
            // document.getElementById('result1').textContent ='Classify Image: ' ;
            document.getElementById('result1a').textContent =predictions + '!';
            document.getElementById('result1b').textContent = 'This is '  ;  
       } })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#recogniseFace').click(uploadAndRecogniseFace);

