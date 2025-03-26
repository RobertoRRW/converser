---
title: Conversation Graph
---
stateDiagram-v2
  Greet --> CollectEmail: Ask initial question
  CollectEmail --> ValidateEmail: Ask for email
  ValidateEmail --> CollectDeviceInfo: Ask about device
  CollectDeviceInfo --> CollectIssueDetails: Get problem details
  CollectDeviceInfo --> CollectDeviceInfo: Keep collecting
  CollectIssueDetails --> ProvideSolutions: Look for solutions
  ProvideSolutions --> CheckSatisfaction
  ProvideSolutions --> ProvideSolutions: Keep trying
  CheckSatisfaction --> Farewell
  CheckSatisfaction --> CheckSatisfaction: Propose more solutions
  Farewell --> [*]


