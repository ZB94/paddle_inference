use std::ffi::CString;

/// 将普通文本转为以`\0`结尾的指针
///
/// - 如果输入字符串**没有**`\0`字符，将以输入字符串为基础创建一个[`CString`]，并返回该值与其指针。使用该指针前需确保返回的[`CString`]存活
/// - 如果输入字符串中**有**`\0`字符，则直接返回`None`和指向字符串的指针。使用该指针前需确保输入字符串存活
pub fn to_c_str(s: &str) -> (Option<CString>, *const i8) {
    CString::new(s)
        .map(|s| {
            let ptr = s.as_ptr();
            (Some(s), ptr)
        })
        .unwrap_or_else(|_| (None, s.as_ptr() as *const _))
}

/// 将 shape 为`[width, height, channel]`的数据转为`[channel, height, width]`
pub fn whc_to_chw<T: Copy>(src: &[T], dst: &mut [T], width: usize, height: usize, channel: usize) {
    let size = width * height;
    for k in 0..channel {
        for y in 0..height {
            let wy = width * y;
            for x in 0..width {
                let src_index = k + channel * (x + wy);
                let dst_index = x + wy + size * k;
                dst[dst_index] = src[src_index];
            }
        }
    }
}

#[test]
fn test_whc_to_chw() {
    let mut src = [0; 4 * 4 * 3];
    for (idx, v) in src.iter_mut().enumerate() {
        *v = idx as u8;
    }

    let dst = [
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 1, 4, 7, 10, 13, 16, 19, 22,
        25, 28, 31, 34, 37, 40, 43, 46, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44,
        47,
    ];

    let mut out = [0; 48];

    whc_to_chw(&src, &mut out, 4, 4, 3);

    assert_eq!(out, dst);
}
