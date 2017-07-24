function vout = vectorSaturationF(v_in,vmin,vmax)
%handles saturation for vector case

normin=norm(v_in);
if normin>=vmax
    vout=vmax*unit_vector(v_in);
elseif normin<=vmin
    vout=vmin*unit_vector(v_in);
else
    vout=v_in;
end

if vmin>=vmax
    fprintf('Error in saturation \n') %called saturation with max as min
end
        
end