package minimax

import (
	"errors"
	"testing"

	"github.com/costa92/llm-agent-contract/llm"
)

func TestBaseRespError_Mapping(t *testing.T) {
	tests := []struct {
		name   string
		status int
		isAuth bool
	}{
		{"ok_zero", 0, false},
		{"auth_1004", 1004, true},
		{"generic_1008", 1008, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := baseRespError("minimax: image", baseResp{StatusCode: tt.status, StatusMsg: "boom"})
			if tt.status == 0 {
				if err != nil {
					t.Fatalf("status 0 must be nil, got %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("status %d must produce error", tt.status)
			}
			if tt.isAuth && !errors.As(err, new(*llm.AuthError)) {
				t.Fatalf("status %d: want AuthError, got %v", tt.status, err)
			}
		})
	}
}
