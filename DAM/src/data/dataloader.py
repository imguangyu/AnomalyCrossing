import torch


def pt_data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms):
    if args.pt_dataset in ['ucf101', 'hmdb51', 'diving48', 'sth_v1']:
        from data.dataset import DataSet as DataSet
    elif args.pt_dataset == 'kinetics':
        from data.video_dataset import VideoDataSet as DataSet
    elif args.pt_dataset == 'dota':
        from data.dota_dsm import DataSet as DataSet
    elif args.pt_dataset == 'bdd':
        from data.bdd_dsm import DataSet as DataSet
    elif args.pt_dataset == 'paper_fold':
        #use the same Dataset as BDD
        from data.bdd_dsm import DataSet as DataSet
    elif args.pt_dataset == 'ucf_crime':
        #use the same Dataset as BDD
        from data.bdd_dsm import DataSet as DataSet
    else:
        Exception("unsupported dataset")
    # print(args.pt_train_list)
    # print(args.pt_root)
    train_dataset = DataSet(args, args.pt_root, args.pt_train_list, num_segments=1, new_length=data_length,
                      stride=args.pt_stride, modality=args.pt_mode, dataset=args.pt_dataset, test_mode=False,
                      image_tmpl=image_tmpl if args.pt_mode in ["rgb", "RGBDiff"]
                      else args.pt_flow_prefix + "{}_{:05d}.jpg", transform=train_transforms,
                      time_thresh=args.pt_time_thresh, video_len_max=args.pt_video_len_max)
    print("training samples:{}".format(train_dataset.__len__()))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.pt_batch_size, shuffle=True,
                                                    num_workers=args.pt_workers, pin_memory=True)
    val_dataset = DataSet(args, args.pt_root, args.pt_val_list, num_segments=1, new_length=data_length,
                          stride=args.pt_stride, modality=args.pt_mode, test_mode=True, dataset=args.pt_dataset,
                          image_tmpl=image_tmpl if args.pt_mode in ["rgb", "RGBDiff"] else args.pt_flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=test_transforms, time_thresh=args.pt_time_thresh, video_len_max=args.pt_video_len_max)
    print("val samples:{}".format(val_dataset.__len__()))
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.pt_batch_size, shuffle=False,
                                                  num_workers=args.pt_workers, pin_memory=True)
    eval_dataset = DataSet(args, args.pt_root, args.pt_val_list, num_segments=1, new_length=data_length,
                          stride=args.pt_stride, modality=args.pt_mode, test_mode=True, dataset=args.pt_dataset,
                          image_tmpl=image_tmpl if args.pt_mode in ["rgb", "RGBDiff"] else args.pt_flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=eval_transforms, full_video=True,
                          time_thresh=args.pt_time_thresh, video_len_max=args.pt_video_len_max)
    print("eval samples:{}".format(eval_dataset.__len__()))
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.pt_batch_size, shuffle=False,
                                                  num_workers=args.pt_workers, pin_memory=True)
    return train_data_loader, val_data_loader, eval_data_loader, train_dataset.__len__(), val_dataset.__len__(), eval_dataset.__len__()


def pt_data_loader_init_ddp(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms):
    if args.pt_dataset in ['ucf101', 'hmdb51', 'diving48', 'sth_v1']:
        from data.dataset import DataSet as DataSet
    elif args.pt_dataset == 'kinetics':
        from data.video_dataset import VideoDataSet as DataSet
    elif args.pt_dataset == 'dota':
        from data.dota_dsm import DataSet as DataSet
    elif args.pt_dataset == 'bdd':
        from data.bdd_dsm import DataSet as DataSet
    else:
        Exception("unsupported dataset")
    # print(args.pt_train_list)
    # print(args.pt_root)

    train_dataset = DataSet(args, args.pt_root, args.pt_train_list, num_segments=1, new_length=data_length,
                      stride=args.pt_stride, modality=args.pt_mode, dataset=args.pt_dataset, test_mode=False,
                      image_tmpl=image_tmpl if args.pt_mode in ["rgb", "RGBDiff"]
                      else args.pt_flow_prefix + "{}_{:05d}.jpg", transform=train_transforms)

    print("training samples:{}".format(train_dataset.__len__()))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=args.pt_batch_size, 
                                                    sampler=train_sampler,
                                                    num_workers=args.pt_workers, 
                                                    pin_memory=True)
    val_dataset = DataSet(args, args.pt_root, args.pt_val_list, num_segments=1, new_length=data_length,
                          stride=args.pt_stride, modality=args.pt_mode, test_mode=True, dataset=args.pt_dataset,
                          image_tmpl=image_tmpl if args.pt_mode in ["rgb", "RGBDiff"] else args.pt_flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=test_transforms)

    print("val samples:{}".format(val_dataset.__len__()))
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.pt_batch_size, 
                                                  sampler=val_sampler,
                                                  num_workers=args.pt_workers, 
                                                  pin_memory=True)
    eval_dataset = DataSet(args, args.pt_root, args.pt_val_list, num_segments=1, new_length=data_length,
                          stride=args.pt_stride, modality=args.pt_mode, test_mode=True, dataset=args.pt_dataset,
                          image_tmpl=image_tmpl if args.pt_mode in ["rgb", "RGBDiff"] else args.pt_flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=eval_transforms, full_video=True)

    print("eval samples:{}".format(eval_dataset.__len__()))
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)

    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, 
                                                  batch_size=args.pt_batch_size, 
                                                  sampler=eval_sampler,
                                                  num_workers=args.pt_workers, 
                                                  pin_memory=True)
    return train_data_loader, val_data_loader, eval_data_loader, train_dataset.__len__(), val_dataset.__len__(), eval_dataset.__len__(), train_sampler, val_sampler, eval_sampler



def ft_data_loader_init(args, data_length, image_tmpl, train_transforms, test_transforms, eval_transforms):
    if args.ft_dataset in ['ucf101', 'hmdb51', 'diving48', 'sth_v1']:
        from data.dataset import DataSet as DataSet
    elif args.ft_dataset == 'kinetics':
        from data.video_dataset import VideoDataSet as DataSet
    else:
        Exception("unsupported dataset")
    train_dataset = DataSet(args, args.ft_root, args.ft_train_list, num_segments=1, new_length=data_length,
                      stride=args.ft_stride, modality=args.ft_mode, dataset=args.ft_dataset, test_mode=False,
                      image_tmpl=image_tmpl if args.ft_mode in ["rgb", "RGBDiff"]
                      else args.flow_prefix + "{}_{:05d}.jpg", transform=train_transforms)
    print("training samples:{}".format(train_dataset.__len__()))
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.ft_batch_size, shuffle=True,
                                                    num_workers=args.ft_workers, pin_memory=True)
    val_dataset = DataSet(args, args.ft_root, args.ft_val_list, num_segments=1, new_length=data_length,
                          stride=args.ft_stride, modality=args.ft_mode, test_mode=True, dataset=args.ft_dataset,
                          image_tmpl=image_tmpl if args.ft_mode in ["rgb", "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=test_transforms)
    print("val samples:{}".format(val_dataset.__len__()))
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.ft_batch_size, shuffle=False,
                                                  num_workers=args.ft_workers, pin_memory=True)
    eval_dataset = DataSet(args, args.ft_root, args.ft_val_list, num_segments=1, new_length=data_length,
                          stride=args.ft_stride, modality=args.ft_mode, test_mode=True, dataset=args.ft_dataset,
                          image_tmpl=image_tmpl if args.ft_mode in ["rgb", "RGBDiff"] else args.ft_flow_prefix + "{}_{:05d}.jpg",
                          random_shift=False, transform=eval_transforms, full_video=True)
    print("eval samples:{}".format(eval_dataset.__len__()))
    eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.ft_batch_size, shuffle=False,
                                                  num_workers=args.ft_workers, pin_memory=True)
    return train_data_loader, val_data_loader, eval_data_loader, train_dataset.__len__(), val_dataset.__len__(), eval_dataset.__len__()
