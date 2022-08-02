featureExtractionNetwork = resnet50;%Burada resnet50 ağı seçilerek ağın geçmişi tutuluyor.
featureLayer = 'activation_40_relu';%aktivasyon katmanımız ise birim basamak katmanı
numClasses = width(randomRcnnTable)-1;
anchorBoxes = [

    29    17
    46    39
   136   116];%anchor box sayesinde görüntü detect taşması engelleniyor.
lgraph = fasterRCNNLayers([224 224 3],numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

imds = imageDatastore(randomRcnnTable.imageFileName);%görüntü veri setimizin tablosu seçiliyor.
blds = boxLabelDatastore(randomRcnnTable(:,2:end));
ds = combine(imds, blds);
ds=transform(ds,@(data)preprocessData(data,[224 224]))%google nete göre 224 224 görüntü boyutu ayarlandı.
options = trainingOptions('sgdm', ...
      'MiniBatchSize', 1, ...
      'InitialLearnRate', 1e-3, ...
      'MaxEpochs', 7, ...
      'VerboseFrequency', 200, ...
      'CheckpointPath', tempdir, ...
      'ExecutionEnvironment','cpu');
load('faster_rcnn_checkpoint__1616__2022_07_21__16_30_34.mat','detector')
lgraph = layerGraph(detector.Network.Layers)
detector2 = trainFasterRCNNObjectDetector(ds,detector.Network.layerGraph , options, ...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1]);%yukarıdaki tüm parametreler ağın eğitimi için gerekli parametrelerdir.(örn:Minibatchsize,epoch,vs.)
%bir tespit edici oluşturuluyor ve eğitilecek olan ağımızı temsil ediyor.
%% 

img = imread('Kopekİsaret\Kopek8.jpg');%eğitim tamamlandıktan sonra(şahsi bilgisayarımda 36 saat sürdü) test için bir köpek görseli seçiliyor
boyut=size(img);
img=imresize(img,[300 300 ]);%grafik kartımızın desteklediği max boyutu deneyerek buluyoruz.
[bbox, score, label] = detect(detector2,img);
detectedImg=img;
for i=1:size(bbox,1)
    ayirtedici=sprintf('%s oran=%f',label(i),score(i));
    detectedImg=insertObjectAnnotation(detectedImg,"rectangle",bbox(i,:),ayirtedici);
end
figure
detectedImg=imresize(detectedImg,boyut(1:2));
imshow(detectedImg)%en sonunda test için seçilen görüntünün köpek cinsinin tespit edilme oranına bakılıyor.