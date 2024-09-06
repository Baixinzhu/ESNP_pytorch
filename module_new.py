# 调用说明
# 这里的hidden_size就是储存池reservoir_size
# self.ESNP = ESNP(input_size=self.batch_size, hidden_size=self.hidden_size, output_size=self.ESNP_nums)
# 输出维度：
# a = self.ESNP(input)
class ESNP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, spectral_radius=0.9, sparsity=0.1, scale_in=0.1,
                 scale_res=0.1, α=0.1, β=1):
        super(ESNP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = α
        self.beta = β
        self.scale_in = scale_in

        # 输入权重矩阵
        self.W_in = nn.Parameter(scale_in * (2 * torch.rand(hidden_size, input_size) - 1), requires_grad=False)

        # 隐状态权重矩阵（稀疏矩阵）---稀疏化水库连接
        W = torch.rand(hidden_size, hidden_size) - 0.5
        mask = torch.rand(hidden_size, hidden_size) > sparsity
        W[mask] = 0
        # 调整谱半径radius，用于控制矩阵的谱半径，以确保网络具有回声状态特性。
        spectral_radius_W = max(abs(torch.eig(W)[0][:, 0]))
        # 水库权重矩阵
        self.W = nn.Parameter(W * (spectral_radius / spectral_radius_W), requires_grad=False)

        # 输出权重的线性变换y=Wx
        self.W_out_u = nn.Linear(input_size, hidden_size)
        self.W_out_h = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 维度假设为 (batch_size, seq_length, input_size)
        batch_size, seq_length, _ = x.size()
        # 调整水库初始化状态,即内部隐藏状态u
        hidden_state = torch.zeros(batch_size, self.hidden_size)
        y = torch.zeros(batch_size, self.hidden_size)
        # 对每个时间步更新水库状态
        for t in range(seq_length):
            u = x[:, t, :]  # 当前时间步的输入x
            # 由于batch_size大小设置不一样，则最后一组时不等于hidden_size则零填充，如没有遇到则可删除以下if代码。
            # if u.shape[0] != self.hidden_size:
            #     mix = self.hidden_size-u.shape[0]
            #     u = F.pad(u,(0,0,0,mix))
            #     hidden_state = F.pad(hidden_state,(0,0,0,mix))
            if (self.W_in.shape[1] != u.shape[1]):
                self.W_in = nn.Parameter(self.scale_in * (2 * torch.rand(self.hidden_size, u.shape[1]) - 1),
                                         requires_grad=False)
                self.W_out_u = nn.Linear(u.shape[1], self.output_size)
            # 原始ESN公式
            # hidden_state = torch.tanh(torch.matmul(u, self.W_in.T) + torch.matmul(hidden_state, self.W.T))
            # 修改的ESNP公式u_t = α * u(t-1) + W * tanh(W_in * x + β * u(t-1))
            # hidden_state = self.alpha * hidden_state + torch.matmul(self.W.T, torch.tanh(torch.matmul(self.W_in.T, u) + self.beta * hidden_state))
            hidden_state = self.alpha * hidden_state + torch.matmul(self.W.T, torch.tanh(
                torch.matmul(self.W_in, u.T) + self.beta * hidden_state))
            # 修改的ESNP公式y_t = W_out_x * x_t + w_out_u * u_t
            y = self.W_out_u(u) + self.W_out_h(hidden_state)
        output = y
        return output