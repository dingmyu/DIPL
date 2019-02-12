load('images.mat');
te_find={'inn/indoor','flea market/indoor','lab classroom','outhouse/outdoor','chemical plant','mineshaft','lake/natural','shoe shop','art school','archive'};
test_class=zeros(1,10);
for j=1:10
for i=1:14340
if strfind(images{i},char(te_find{1,j}))
test_class(j)=i;
i=14340;
end
end
end
