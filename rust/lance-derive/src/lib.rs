// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Identifiers introduced by this macro are created with a mixed-site span so
/// they resolve at the definition site for local variables. Without that, a
/// user field literally named `__context` (an enum variant field is bound by
/// name in the generated pattern) would shadow the `Context` parameter.
fn internal_ident(name: &str) -> syn::Ident {
    syn::Ident::new(name, proc_macro2::Span::mixed_site())
}

/// Derive macro for the `DeepSizeOf` trait.
///
/// Generates an implementation that sums the `deep_size_of_children` of all
/// fields (for structs) or the active variant's fields (for enums).
#[proc_macro_derive(DeepSizeOf)]
pub fn derive_deep_size_of(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;
    let context_ident = internal_ident("__lance_context");

    // Add DeepSizeOf bounds to all type parameters
    let mut bounded_generics = generics.clone();
    for param in &mut bounded_generics.params {
        if let syn::GenericParam::Type(ref mut type_param) = *param {
            type_param
                .bounds
                .push(syn::parse_quote!(lance_core::deepsize::DeepSizeOf));
        }
    }
    let (impl_generics, _, where_clause) = bounded_generics.split_for_impl();
    let (_, ty_generics, _) = generics.split_for_impl();

    let body = match &input.data {
        Data::Struct(data) => generate_struct_body(&data.fields, &context_ident),
        Data::Enum(data) => {
            let arms: Vec<_> = data
                .variants
                .iter()
                .map(|variant| {
                    let variant_ident = &variant.ident;
                    match &variant.fields {
                        Fields::Unit => {
                            quote! { Self::#variant_ident => 0 }
                        }
                        Fields::Unnamed(fields) => {
                            let bindings: Vec<_> = (0..fields.unnamed.len())
                                .map(|i| internal_ident(&format!("__lance_field_{}", i)))
                                .collect();
                            let sum = bindings.iter().map(|b| {
                                quote! { lance_core::deepsize::DeepSizeOf::deep_size_of_children(#b, #context_ident) }
                            });
                            quote! {
                                Self::#variant_ident(#(#bindings),*) => {
                                    0 #(+ #sum)*
                                }
                            }
                        }
                        Fields::Named(fields) => {
                            // Bind each field to a macro-owned identifier instead of
                            // using a shorthand pattern, so the user's field name never
                            // becomes a local variable in the generated code.
                            let field_bindings: Vec<_> = fields
                                .named
                                .iter()
                                .enumerate()
                                .map(|(i, field)| {
                                    let field_name = field.ident.as_ref().expect("named field");
                                    let binding = internal_ident(&format!("__lance_field_{}", i));
                                    (field_name, binding)
                                })
                                .collect();
                            let pattern_fields = field_bindings
                                .iter()
                                .map(|(field_name, binding)| quote! { #field_name: #binding });
                            let sum = field_bindings.iter().map(|(_, binding)| {
                                quote! { lance_core::deepsize::DeepSizeOf::deep_size_of_children(#binding, #context_ident) }
                            });
                            quote! {
                                Self::#variant_ident { #(#pattern_fields),* } => {
                                    0 #(+ #sum)*
                                }
                            }
                        }
                    }
                })
                .collect();
            if arms.is_empty() {
                // `self` is `&Self`, and references are always considered inhabited,
                // so `match self {}` does not type-check for a zero-variant enum.
                // Matching the place itself keeps the empty match valid.
                quote! {
                    match *self {}
                }
            } else {
                quote! {
                    match self {
                        #(#arms),*
                    }
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new_spanned(&input, "DeepSizeOf cannot be derived for unions")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics lance_core::deepsize::DeepSizeOf for #name #ty_generics #where_clause {
            fn deep_size_of_children(&self, #context_ident: &mut lance_core::deepsize::Context) -> usize {
                #body
            }
        }
    };

    TokenStream::from(expanded)
}

fn generate_struct_body(fields: &Fields, context_ident: &syn::Ident) -> proc_macro2::TokenStream {
    match fields {
        Fields::Named(fields) => {
            let field_sizes = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote! { lance_core::deepsize::DeepSizeOf::deep_size_of_children(&self.#name, #context_ident) }
            });
            quote! { 0 #(+ #field_sizes)* }
        }
        Fields::Unnamed(fields) => {
            let field_sizes = (0..fields.unnamed.len()).map(|i| {
                let index = syn::Index::from(i);
                quote! { lance_core::deepsize::DeepSizeOf::deep_size_of_children(&self.#index, #context_ident) }
            });
            quote! { 0 #(+ #field_sizes)* }
        }
        Fields::Unit => {
            quote! { 0 }
        }
    }
}
