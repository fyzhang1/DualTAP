#!/usr/bin/env python3
"""
测试不同应用的字段提取
"""

from privacy_metrics import PrivacyMetrics

def test_field_extraction():
    """测试三种不同应用的字段提取"""
    
    metrics = PrivacyMetrics()
    
    print("=" * 60)
    print("测试隐私字段提取（支持多应用格式）")
    print("=" * 60)
    
    # 测试1: Amazon 格式 (name + emailAddress)
    print("\n【测试1: Amazon 格式】")
    amazon_true = "name: BRIttney WagNER, emailAddress: joelyamada3717@msn.edu"
    amazon_pred = "name: Brittney Wagner, emailAddress: joelyamada3717@msn.edu"
    
    print(f"True answer: {amazon_true}")
    print(f"Pred answer: {amazon_pred}")
    
    true_fields = metrics.extract_all_fields(amazon_true)
    pred_fields = metrics.extract_all_fields(amazon_pred)
    
    print(f"提取的真实字段: {true_fields}")
    print(f"提取的预测字段: {pred_fields}")
    
    result = metrics.evaluate_privacy_leakage(amazon_pred, amazon_true, threshold=0.8)
    print(f"评估结果: leakage_rate={result['leakage_rate']}, protection_rate={result['protection_rate']}")
    print(f"详细匹配: {result['detailed_matches']}")
    
    # 测试2: Ins 格式 (name + Username)
    print("\n【测试2: Ins 格式 (Username是电话号码)】")
    ins_true = "name: CaROlIne mCinTyRe, Username: +27 90 989 6629"
    ins_pred = "name: Caroline McIntyre, Username: +27 90 989 6629"
    
    print(f"True answer: {ins_true}")
    print(f"Pred answer: {ins_pred}")
    
    true_fields = metrics.extract_all_fields(ins_true)
    pred_fields = metrics.extract_all_fields(ins_pred)
    
    print(f"提取的真实字段: {true_fields}")
    print(f"提取的预测字段: {pred_fields}")
    
    result = metrics.evaluate_privacy_leakage(ins_pred, ins_true, threshold=0.8)
    print(f"评估结果: leakage_rate={result['leakage_rate']}, protection_rate={result['protection_rate']}")
    print(f"详细匹配: {result['detailed_matches']}")
    
    # 测试3: Email 格式 (只有 emailAddress)
    print("\n【测试3: Email 格式 (只有emailAddress)】")
    email_true = "emailAddress: ramesh_petrov@msn.net"
    email_pred = "emailAddress: ramesh_petrov@msn.net"
    
    print(f"True answer: {email_true}")
    print(f"Pred answer: {email_pred}")
    
    true_fields = metrics.extract_all_fields(email_true)
    pred_fields = metrics.extract_all_fields(email_pred)
    
    print(f"提取的真实字段: {true_fields}")
    print(f"提取的预测字段: {pred_fields}")
    
    result = metrics.evaluate_privacy_leakage(email_pred, email_true, threshold=0.8)
    print(f"评估结果: leakage_rate={result['leakage_rate']}, protection_rate={result['protection_rate']}")
    print(f"详细匹配: {result['detailed_matches']}")
    
    # 测试4: 部分泄露的情况
    print("\n【测试4: 部分泄露 (名字被保护，email泄露)】")
    partial_true = "name: John Smith, emailAddress: john@example.com"
    partial_pred = "emailAddress: john@example.com"  # 只泄露了email
    
    print(f"True answer: {partial_true}")
    print(f"Pred answer: {partial_pred}")
    
    result = metrics.evaluate_privacy_leakage(partial_pred, partial_true, threshold=0.8)
    print(f"评估结果: leakage_rate={result['leakage_rate']}, protection_rate={result['protection_rate']}")
    print(f"详细匹配: {result['detailed_matches']}")
    
    # 测试5: Username 在 ins 数据中的实际情况
    print("\n【测试5: Ins Username 字段提取测试】")
    ins_formats = [
        "name: taMmY pINEdA, Username: 0771 620 920",
        "name: Mr. cAMeROn RoBINsoN, Username: (49) 93463-8352",
        "name: jAsminE RodrIgueZ, Username: +91-69477 96124",
    ]
    
    for test_text in ins_formats:
        print(f"\n  输入: {test_text}")
        fields = metrics.extract_all_fields(test_text)
        print(f"  提取结果:")
        for k, v in fields.items():
            if v:
                print(f"    {k}: {v}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_field_extraction()




