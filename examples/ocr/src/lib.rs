use once_cell::sync::Lazy;
use opencv::{
    core::{
        copy_make_border, Point2i, Rect, Size, Vector, BORDER_CONSTANT, BORDER_ISOLATED, CV_32FC1,
        CV_8UC1,
    },
    imgproc::{
        bounding_rect, find_contours, resize, CHAIN_APPROX_SIMPLE, INTER_LINEAR, RETR_EXTERNAL,
    },
    prelude::{Mat, MatTraitConst, MatTraitConstManual, MatTraitManual},
};
use paddle_inference::{ctypes::PlaceType, utils::hwc_to_chw, Predictor};

const LABEL_ALL: &'static str = include_str!("../resource/ppocr_keys_v1.txt");
pub const LABEL_LIST: Lazy<Vec<&'static str>> = Lazy::new(|| LABEL_ALL.lines().collect());

/// 识别文本行
pub fn recognize(
    model: &Predictor,
    image: &Mat,
    roi_list: &[Rect],
    batch_size: usize,
) -> Result<Vec<String>, opencv::Error> {
    if roi_list.is_empty() {
        return Ok(vec![]);
    }

    if roi_list.len() > batch_size {
        let mut all = vec![];
        for l in roi_list.chunks(batch_size) {
            all.append(&mut recognize(model, image, l, batch_size)?);
        }

        return Ok(all);
    }

    const H: i32 = 32;

    // 计算行宽
    let bw = roi_list
        .iter()
        .map(|r| (r.width as f32 / r.height as f32).ceil() as i32)
        .max()
        .unwrap();
    let width = bw * H;
    let bh = roi_list.len() as i32;

    // 获取模型输入
    let input = model.input("x");
    input.reshape(&[bh, 3, H, width]);
    let buff = input.as_mut_slice_f32(PlaceType::Cpu).unwrap();

    // 将识别区域转为模型输入格式并复制到模型输入中
    let size = 3usize * H as usize * width as usize;
    for (rect, buff) in roi_list.iter().zip(buff.chunks_mut(size)) {
        let mut roi = Mat::default();
        copy_make_border(
            &Mat::roi(image, *rect)?,
            &mut roi,
            0,
            0,
            0,
            bw * rect.height - rect.width,
            BORDER_CONSTANT | BORDER_ISOLATED,
            Default::default(),
        )?;

        let mut roi_resize = Mat::default();
        resize(
            &roi,
            &mut roi_resize,
            Size::new(width, H),
            0.0,
            0.0,
            INTER_LINEAR,
        )?;

        roi_resize
            .reshape(1, 1)?
            .convert_to(&mut roi, CV_32FC1, 1.0 / 255.0, 0.0)?;
        hwc_to_chw(roi.data_typed()?, buff, width as usize, H as usize, 3);
    }

    if model.run() {
        let names = model.output_names();
        let out = model.output(&names.get(0).unwrap());
        let size = out.shape();
        let chunks_size = size[2] as usize;
        let line_chunks_size = size[1] as usize * chunks_size;

        let mut data = vec![0.0f32; size.iter().fold(1usize, |s, v| s * *v as usize)];
        out.copy_to_f32(&mut data);

        let range = 1..LABEL_LIST.len();
        let r = data
            .chunks(line_chunks_size)
            .map(|line| {
                line.chunks(chunks_size)
                    .filter_map(|c| {
                        c.iter()
                            .enumerate()
                            .max_by_key(|(_, v)| (*v * 10000.0) as usize)
                            .and_then(|(k, _)| range.contains(&k).then(|| LABEL_LIST[k - 1]))
                    })
                    .collect::<Vec<_>>()
                    .join("")
            })
            .collect();

        Ok(r)
    } else {
        Err(opencv::Error::new(-2002, "识别文本行失败"))
    }
}

/// 匹配文本行位置
pub fn detect(model: &Predictor, image: &Mat) -> Result<Vec<Rect>, opencv::Error> {
    // 获取输入图片大小
    let (w, h) = image.size().map(|s| (s.width, s.height))?;
    let width = w as usize;
    let height = h as usize;

    // 将图片大小扩展到32的倍数
    const S: f32 = 32.0;
    let in_w = ((width as f32 / S).ceil() * S) as i32;
    let in_h = ((height as f32 / S).ceil() * S) as i32;

    let mut in_mat = Mat::default();
    copy_make_border(
        &image,
        &mut in_mat,
        0,
        in_h - h,
        0,
        in_w - w,
        BORDER_CONSTANT | BORDER_ISOLATED,
        Default::default(),
    )?;

    // 获取模型输入
    let input = model.input("x");
    input.reshape(&[1, 3, in_h, in_w]);
    let buff = input.as_mut_slice_f32(PlaceType::Cpu).unwrap();

    // 将输入图片转为模型输入格式并复制到模型输入中
    let mut det_mat_f32 = Mat::default();
    in_mat
        .reshape(1, 1)?
        .convert_to(&mut det_mat_f32, CV_32FC1, 1.0 / 255.0, 0.0)?;

    hwc_to_chw(
        det_mat_f32.data_typed()?,
        buff,
        in_w as usize,
        in_h as usize,
        3,
    );

    // 运行模型
    if model.run() {
        // 获取模型输出并复制到内存中
        let names = model.output_names();
        let out = model.output(&names.get(0).unwrap());
        let mut img_f32 =
            Mat::new_size_with_default(Size::new(in_w * in_h, 1), CV_32FC1, Default::default())?;
        out.copy_to_f32(img_f32.data_typed_mut()?);

        // 将输出转为Mat
        let mut img = Mat::default();
        img_f32
            .reshape(1, in_h)?
            .convert_to(&mut img, CV_8UC1, 255.0, 0.0)?;

        // 查找轮廓
        let mut points: Vector<Vector<Point2i>> = Default::default();
        find_contours(
            &img,
            &mut points,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            Default::default(),
        )?;

        // 提取、筛选轮廓
        let l = points
            .into_iter()
            .map(|l| bounding_rect(&l))
            .filter_map(|r| {
                r.ok().and_then(|mut rect| {
                    let offset = rect.height;
                    rect.x = (rect.x - offset).clamp(0, w);
                    rect.width = (rect.width + offset * 2).min(w - rect.x);
                    rect.y = (rect.y - offset).clamp(0, h);
                    rect.height = (rect.height + offset * 2).min(h - rect.y);

                    if rect.width < 4 || rect.height < 8 {
                        None
                    } else {
                        Some(rect)
                    }
                })
            })
            .collect();

        Ok(l)
    } else {
        Err(opencv::Error::new(-2001, "定位文本行是失败"))
    }
}
