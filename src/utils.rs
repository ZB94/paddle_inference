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
