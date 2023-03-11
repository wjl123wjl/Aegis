for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            tmp=m.weight.detach().clone()
            tmp_n=tmp.cpu().detach().numpy()
            tmp_uint8 = tmp_n.astype("int8")
            #print("tmp_uint8:", tmp_uint8[-1])
            tmp_uint8[(tmp_uint8>=-32) & (tmp_uint8<=0)] = tmp_uint8[(tmp_uint8>=-32) & (tmp_uint8<=0)] ^ 0x80
            tmp_uint8[(tmp_uint8<=32) & (tmp_uint8>=0)] = tmp_uint8[(tmp_uint8<=32) & (tmp_uint8>=0)] ^ 0x80
            
            min_v= min_pos * m.step_size
            print("min v:", min_v, m.step_size, min_pos)

            tmp_n = tmp_uint8.astype('float32')
            
            an=torch.tensor(tmp_n).cuda()
            
            m.weight.data = torch.tensor(tmp_n).cuda()
            #m.__reset_stepsize__()
            m.inf_with_weight =False
    return 