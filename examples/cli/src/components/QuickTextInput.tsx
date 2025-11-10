import React from 'react';
import TextInput from 'ink-text-input';

interface QuickTextInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: (value: string) => void;
  placeholder?: string;
  focus?: boolean;
}

/**
 * Simple TextInput wrapper
 * Note: Clear functionality (Cmd/Ctrl+Backspace) should be handled
 * in parent component's useInput to avoid conflicts with TextInput's internal handlers
 */
export const QuickTextInput: React.FC<QuickTextInputProps> = ({
  value,
  onChange,
  onSubmit,
  placeholder,
  focus = true
}) => {
  return (
    <TextInput
      value={value}
      onChange={onChange}
      onSubmit={onSubmit}
      placeholder={placeholder}
      focus={focus}
    />
  );
};
