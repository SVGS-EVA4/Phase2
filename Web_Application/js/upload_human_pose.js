function uploadAndGetPose(){
    var fileInput = document.getElementById('fileUpload').files;
    if(!fileInput.length){
        return alert('Please choose file')
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
        url:'https://xxxxxxx.execute-api.ap-south-1.amazonaws.com/dev/human_pose_estimation',
        data: formData,
        processData: false,
        
        contentType: false,
        
   
        mimeType:'multipart/form-data',


    })
    .done(function(response){
        console.log('hello');
        if (((JSON.parse(response)).Status) =='IncorrectInput'){
            document.getElementById('result').src = '/';
            document.getElementById('error').textContent = 'No person detected';
        }
        else{
            document.getElementById('error').textContent = '';
            document.getElementById('result').src = 'data:image/jpeg;base64,'+(JSON.parse(response)).ImageBytes;
            document.getElementById('result').style.display = 'inherit'
        }
    })
    .fail(function(){
        alert('Error during prediction');
    });
};

$('#btnUpload').click(uploadAndGetPose);