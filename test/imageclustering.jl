using  DPMM, Images


img = load(download("http://docs.opencv.org/3.1.0/water_coins.jpg"));
img = Gray.(img)

imghw = reshape(Float64.(channelview(img)), 1, :)
X = zeros(2,length(img))
inds = reshape(CartesianIndices(img),:)
X[1,:] .= map(i->i[2],inds)
X[2,:] .= map(i->i[1],inds)

labels = fit(imghw; T=10000, α=5.0, scene=setup_scene(X))


img = load(download("https://juliaimages.org/latest/assets/segmentation/flower.jpg"));
imghw = reshape(Float64.(channelview(img)), 3, :)
X = zeros(2,length(img))
inds = reshape(CartesianIndices(img),:)
X[1,:] .= map(i->i[2],inds)
X[2,:] .= map(i->i[1],inds)
labels = fit(imghw; T=100, α=5.0, scene=setup_scene(X))
